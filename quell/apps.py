import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchapp as ta
import torchvision
from fastai.callback.tracker import SaveModelCallback
from fastai.data.block import DataBlock, TransformBlock
from fastai.data.core import DataLoaders
from fastai.data.load import DataLoader
from fastai.data.transforms import (ColSplitter, DisplayedTransform, FuncSplitter,
                                  IndexSplitter, ToTensor)
from fastai.learner import load_learner
from fasttransform import Pipeline
from rich.box import SIMPLE
from rich.console import Console
from rich.progress import track
from rich.table import Table
from torchapp.util import call_func

from .callbacks import IdentityCallback, NoiseIncludeCallBack
from .metrics import (L1_full, L2_full, QuellLoss, masked_psnr,
                     masked_psnr_full, ssim, ssim_full)
from .models import ConvLayersModel, Identity, Unet
from .transforms import (CropItem, CropListGetter, KeepCalcifications, Rescale,
                        VolumeBlock, beta_to_12bit, read_crop_item_denan,
                        read_volume, thick_axial, unsqueeze, write_dicom,
                        write_volume)
from .utils import (distance_to_boundary, generate_intervals,
                   generate_overlapping_intervals)

console = Console()


class Quell(ta.TorchApp):
    """
    An app to denoise CT scan images.
    """
    def dataloaders(
        self,
        csv:Path = ta.Param(
            help="A CSV input file with a column 'x' with the paths to input images,"
                " 'y' with paths to target images "
                "and 'partition' with the cross-validation partition index."
        ), 
        base_dir: Path = ta.Param(
            default=None, 
            help="The base directory for images with relative paths. "
                "If not given, then it is relative to the csv directory."
        ),
        preprocessed_dir:Path = ta.Param(
            default=None, 
            help="A directory to store preprocessed input files."
        ),
        validation_partition:int = ta.Param(
            default=0,
            help="The partition to use for validation in this run."
        ), 
        batch_size:int = ta.Param(default=1, help="The batch size."),
        size_i:int=256,
        size_j:int=256,
        size_k:int=64,
        debug:bool=False,
        max_items:int=None,
        cache_memory:bool=ta.Param(default=False, help="Whether or not to cache all data in memory. Requires large memory. Reduces number of disk reads."),
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which Quell uses in training and prediction.

        Returns:
            DataLoaders: The DataLoaders object.

        """
        df = pd.read_csv(csv)

        assert ('validation' in df or 'partition' in df)
        if 'validation' not in df:
            df['validation'] = df['partition'] == validation_partition

        base_dir = base_dir or Path(csv).parent
        base_dir = Path(base_dir)

        # Preprocess the input files if necessary
        assert preprocessed_dir is not None
        preprocessed_dir = Path(preprocessed_dir)

        indexes_to_keep = []
        for i, row in df.iterrows():
            x_preprocessed_path, y_preprocessed_path = preprocessed_dir/f"{row['x']}", preprocessed_dir/f"{row['y']}"

            if x_preprocessed_path.exists() and y_preprocessed_path.exists():
                indexes_to_keep.append(i)
            else:
                print(f"Data not found in {x_preprocessed_path} or {y_preprocessed_path}")

        df = df.iloc[indexes_to_keep]

        inputs = []
        outputs = []
        validation_indexes = []
        cache = dict()
        for _, row in track(df.iterrows(), description="Preprocessing...", total=len(df)):
            y = read_volume(preprocessed_dir/row['y'])

            # load all into cache
            if cache_memory:
                cache[preprocessed_dir/row['y']] = y
                cache[preprocessed_dir/row['x']] = read_volume(preprocessed_dir/row['x'])

            # Problem in preprocessed files
            if y.shape[2] == 0:
                continue

            for start_i, end_i in generate_intervals(y.shape[0], size_i):
                for start_j, end_j in generate_intervals(y.shape[1], size_j):
                    for start_k, end_k in generate_intervals(y.shape[2], size_k):
                        coords = dict(
                            start_i=start_i,
                            end_i=end_i,
                            start_j=start_j,
                            end_j=end_j,
                            start_k=start_k,
                            end_k=end_k,
                        )
                        cropped = y[start_i:end_i,start_j:end_j,start_k:end_k]
                        if not cropped.isnan().all():
                            if row['validation']:
                                validation_indexes.append(len(inputs))
                            inputs.append( CropItem(path=preprocessed_dir/row['x'],**coords) )
                            outputs.append( CropItem(path=preprocessed_dir/row['y'],**coords) )



            if debug and len(validation_indexes) > 2 and len(inputs) > len(validation_indexes) + batch_size:
                break

        datablock = DataBlock(
            blocks=[DataBlock, DataBlock],
            get_x=CropListGetter(inputs, denan=True, cache=cache),
            get_y=CropListGetter(outputs, denan=True, cache=cache), # if this is False then the weights aren't updated.
            splitter=IndexSplitter(validation_indexes),
        )

        dataloaders = datablock.dataloaders(
            list(range(len(inputs))), 
            bs=batch_size,
        )

        return dataloaders

    def monitor(self):
        # cancel TorchApp's default save model callback
        return False

    def extra_callbacks(self, 
                        save_interval:int = ta.Param( 
                            default=0, 
                            help="Epoch interval to save the model. Set to 0 to only save the best."
                            )):
        callbacks = [NoiseIncludeCallBack()]

        # model saving callbacks, best_save call back must be after interval_save
        if save_interval > 0:
            callbacks.append(SaveModelCallback(every_epoch=save_interval))

        # best model
        callbacks.append(SaveModelCallback(monitor="valid_loss"))
        return callbacks

    def loss_func(
        self, 
        smooth_l1_factor:float=1.0,
        residual_ratio_factor:float=0.0, 
        ssim_loss_factor:float=0.0,
    ):
        return QuellLoss(
            smooth_l1_factor=smooth_l1_factor,
            residual_ratio_factor=residual_ratio_factor, 
        )

    def model(
        self, 
        in_channels:int=1,
        pretrained:Path=None,
        filters:int=32,
        kernel:int=3,
        layers:int=3
    ):
        self.in_channels = in_channels
        if pretrained:
            learner = load_learner(pretrained)
            return learner.model            

        return Unet(
            in_channels=self.in_channels, 
            filters=filters, 
            kernel_size=kernel,
            layers=layers)
    
    def metrics(self):
        return [
            masked_psnr_full,
            ssim_full,
            L1_full,
            L2_full, 
            masked_psnr,
            ssim,
        ]

    def inference_dataloader(
        self, 
        learner, 
        batch_size:int = 1,
        item:Path = None, 
        size_i:int=256,
        size_j:int=256,
        size_k:int=64,
        overlap:int=0,
        overlap_i:int=0,
        overlap_j:int=0,
        overlap_k:int=0,
        **kwargs
    ):  

        # Set the size of the overlap.
        # Can this be saved in the learner or extracted from in the model?
        overlap_i = overlap_i or overlap
        overlap_j = overlap_j or overlap
        overlap_k = overlap_k or overlap
        
        self.input = read_volume(item)
        if isinstance(self.input, np.ndarray):
            # read in 32bit beta scale tif stack
            self.input = torch.Tensor(self.input)
            # need to convert to model scale
            self.input = Rescale().encodes(self.input)
        self.crops = []
        self.shape = self.input.shape

        # pad input with at least 1 voxel
        # if input shape is smaller than size_i, size_j, size_k then pad input with nans
        sizes = [size_i, size_j, size_k]
        pad_sizes = [max(1, size - self.shape[i]) for i, size in enumerate(sizes)]
        pad_dims = [(1, pad) for pad in reversed(pad_sizes)]
        self.input = torch.nn.functional.pad(self.input, sum(pad_dims, ()), mode="constant", value=float('nan'))
        self.pad_sizes = pad_sizes
        self.crop_shape = (size_i, size_j, size_k)

        for start_i, end_i in generate_overlapping_intervals(self.input.shape[0], size_i, overlap_i):
            for start_j, end_j in generate_overlapping_intervals(self.input.shape[1], size_j, overlap_j):
                for start_k, end_k in generate_overlapping_intervals(self.input.shape[2], size_k, overlap_k):
                    coords = dict(
                        start_i=start_i,
                        end_i=end_i,
                        start_j=start_j,
                        end_j=end_j,
                        start_k=start_k,
                        end_k=end_k,
                    )
                    cropped = self.input[start_i:end_i,start_j:end_j,start_k:end_k]
                    if not cropped.isnan().all():
                        self.crops.append( CropItem(path=item,**coords) )

        self.dataloader = DataLoader(
            dataset=CropListGetter(self.crops, denan=True, cache={item:self.input}),
            batch_size=batch_size,
            after_item=Pipeline( [read_crop_item_denan, unsqueeze, ToTensor] )
        )
        
        return self.dataloader
    
    def process_results(self, results):
        # weight the voxels in the crops by the distance from the pixel to the boundary when stitching them back together
        weight = distance_to_boundary(*self.crop_shape)

        predicted_residual = torch.zeros(self.input.shape, dtype=torch.float32)
        summed_weights = torch.zeros(self.input.shape, dtype=int)

        for crop, result in track(zip(self.crops, results[0]), total=len(self.crops), description="Stitching output into single volume:"):
            predicted_residual[crop.start_i:crop.end_i,crop.start_j:crop.end_j,crop.start_k:crop.end_k] += result.squeeze(dim=0) * weight
            summed_weights[crop.start_i:crop.end_i,crop.start_j:crop.end_j,crop.start_k:crop.end_k] += weight
        
        # divide by the weights
        non_zero_voxels = summed_weights > 0
        predicted_residual[non_zero_voxels] /= summed_weights[non_zero_voxels]
        predicted_residual[~non_zero_voxels] = math.nan

        prediction = self.input + predicted_residual
        
        # depad the prediction and input
        prediction = prediction[1:-self.pad_sizes[0],
                                1:-self.pad_sizes[1],
                                1:-self.pad_sizes[2]]
        self.input = self.input[1:-self.pad_sizes[0],
                                1:-self.pad_sizes[1],
                                1:-self.pad_sizes[2]]

        return prediction

    def output_results(
        self, 
        results, 
        output: Path = ta.Param(None, help="The location of the output file. path/to/dir/prefix.tif to save as tiff sequence."),
        thick_slice: bool = ta.Param(False, help="If True, then it outputs axial thick slices (30 slice over, 15 slice step, 12-bit encoding, dicom output)."),
        half_precision: bool = ta.Param(False, help="The precision of the output file. If True, then it outputs in 16-bit floats, otherwise it uses 32-bit floats."),
        calcifications: float=ta.Param(default=None, help="Threshold value to keep calcifications (extreme values) from input to output image in beta scale, e.g. 2e-10"),
        rescale: bool=ta.Param(default=True, help="Rescale the output to beta values. Cannot be used with half-precision."),
        dcm: bool=ta.Param(default=False, help="If True, then it outputs volume as DICOM slices."),
        dcm_name: str=ta.Param(default="NAME", help="The patient name and ID to store in DICOM file."),
        dcm_sex: str=ta.Param(default="O", help="The patient sex either (O, M, F). To store in DICOM file"),
        dcm_details: str=ta.Param(default="DETAILS", help="Study details to store in DICOM file"),
        dcm_laterality: str=ta.Param(default="R", help="The laterality (L or R) of the image to store in DICOM file"),
        **kwargs,
    ):

        results = self.process_results(results)
        if half_precision:
            # convert to 16bit
            results=results.type(torch.float16)

        if calcifications is not None:
            calcifications = Rescale().encodes(torch.tensor(calcifications)).item()
            results = KeepCalcifications(manual_threshold=calcifications).encodes(results, self.input.type(results.dtype))

        if rescale:
            # rescale to beta values
            try:
                results = Rescale().decodes(results)
                console.print("Rescaled to beta values.", style="green")
            except TypeError:
                console.print("Rescaling to beta values is not compatible with half precision. Will not rescale.", style="red")

        if thick_slice:
            results = thick_axial(results)

        results = results.nan_to_num(nan=0.0)
        assert output is not None

        if dcm:
            # convert to 12bit
            results = beta_to_12bit().encodes(results)
        
            # save as dicom
            write_dicom(results, output, file_prefix='ax', dcm_name=dcm_name, dcm_sex=dcm_sex, dcm_laterality=dcm_laterality, dcm_details=dcm_details)
        else:
            write_volume(results, output)
        console.print(f"Output volume saved to '{output}'")
        return output

    def pretrained_location(
        self,
    ) -> str:
        raise NotImplementedError()

    def validate(
        self,
        gpu: bool = ta.Param(True, help="Whether or not to use a GPU for processing if available."),
        pretrained:Path=None,
        result_csv:Path=ta.Param(None, help="CSV file to store results in. Appends if it already exists."),
        eval_name:str=ta.Param(None, help="Entry name for storing in result-csv."),
        **kwargs,
    ):
        # Check if CUDA is available
        gpu = gpu and torch.cuda.is_available()

        # load exported learner
        try:
            learner = load_learner(pretrained, cpu=not gpu)
        except Exception:
            import dill
            learner = load_learner(pretrained, cpu=not gpu, pickle_module=dill)

        # load dataloaders
        dataloaders = call_func(self.dataloaders, **kwargs)

        table = Table(title="Validation", box=SIMPLE)

        # need to readd callbacks that arn't stored in the learner
        extra_callbacks = call_func(self.extra_callbacks, **kwargs)
        values = learner.validate(dl=dataloaders.valid,cbs=extra_callbacks)
        names = [learner.recorder.loss.name] + [metric.name for metric in learner.metrics]
        result = {name: value for name, value in zip(names, values)}

        table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        for name, value in result.items():
            table.add_row(name, str(value))

        console.print(table)

        # save to file
        if result_csv is not None:
            if not isinstance(result_csv, Path):
                result_csv = Path(result_csv)
                
            if eval_name is None:
                eval_name = 'unnamed'

            df = pd.DataFrame(result, index=[eval_name])

            if Path(result_csv).exists():
                ogdf = pd.read_csv(result_csv, index_col=0)
                df = pd.concat([ogdf, df])
            
            df.to_csv(result_csv, index=True)

        return result
    
class QuellIdentity(Quell):
    '''
    Identity model.
    Use this for functionality without affecting images.
    '''
    
    def process_results(self, results):
        return results
    
    def model(self):
        return Identity()

    def extra_callbacks(self):
        return [NoiseIncludeCallBack, IdentityCallback()]

    def __call__(
        self, 
        gpu: bool = ta.Param(False, help="Whether or not to use a GPU for processing if available."), 
        item:Path = None, 
        **kwargs
        ):
        # Check if CUDA is available
        gpu = gpu and torch.cuda.is_available()

        # Create a dataloader for inference
        results = read_volume(item)

        # default at fp32
        results = results.type(torch.float32)
        # Output results
        output_results = ta.util.call_func(self.output_results, results, **kwargs)
        return output_results if output_results is not None else results
    
    def validate(
        self,
        gpu: bool = ta.Param(True, help="Whether or not to use a GPU for processing if available."),
        pretrained:Path=None,
        result_csv:Path=ta.Param(None, help="CSV file to store results in. Appends if it already exists."),
        eval_name:str=ta.Param(None, help="Entry name for storing in result-csv."),
        **kwargs,
    ):
        # Check if CUDA is available
        gpu = gpu and torch.cuda.is_available()
        learner = call_func(self.learner, **kwargs)

        # Move to GPU if available
        if gpu:
            learner.model = learner.model.cuda()
        table = Table(title="Validation", box=SIMPLE)

        # need to read callbacks that arn't stored in the learner
        extra_callbacks = call_func(self.extra_callbacks, **kwargs)
        # learner already has a validation dataloader
        values = learner.validate(cbs=extra_callbacks)
        names = [learner.recorder.loss.name] + [metric.name for metric in learner.metrics]
        result = {name: value for name, value in zip(names, values)}

        table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        for name, value in result.items():
            table.add_row(name, str(value))

        console.print(table)

        # save to file
        if result_csv is not None:
            if not isinstance(result_csv, Path):
                result_csv = Path(result_csv)
                
            if eval_name is None:
                eval_name = 'unnamed'

            df = pd.DataFrame(result, index=[eval_name])

            if Path(result_csv).exists():
                ogdf = pd.read_csv(result_csv, index_col=0)
                df = pd.concat([ogdf, df])
            
            df.to_csv(result_csv, index=True)

        return result
    

