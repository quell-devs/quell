from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
import tifffile
from fastai.data.block import TransformBlock
from fastai.vision.core import PILImageBW
from fasttransform import DisplayedTransform
from PIL import Image
from rich.progress import track
from scipy import ndimage
from skimage import io
from skimage.transform import resize as skresize
from tqdm import tqdm

import nrrd
import nrrd.reader
import nrrd.writer

# Add new type mappings
nrrd.writer._TYPEMAP_NUMPY2NRRD.update({
    'f2': 'float16',
})
nrrd.reader._TYPEMAP_NRRD2NUMPY.update({
    'float16': 'f2',
})


@dataclass
class CropItem():
    path:Path
    start_i:int
    end_i:int
    start_j:int
    end_j:int
    start_k:int
    end_k:int
    
    def read(self):
        return read_volume(self.path)
    
    def crop(self, tensor):
        return tensor[self.start_i:self.end_i,self.start_j:self.end_j,self.start_k:self.end_k]


def read_crop_item(item:CropItem, denan:bool=True):
    assert isinstance(item, CropItem)
    cropped = item.read().unsqueeze(dim=0)
    if denan:
        cropped = cropped.nan_to_num(nan=-1.0)
    return cropped


def read_crop_item_denan(item:CropItem):
    return read_crop_item(item, denan=True)


@dataclass
class CropListGetter(DisplayedTransform):
    crops:list[CropItem]
    denan:bool
    cache:dict[Path, torch.Tensor] = field(default_factory=dict)

    def __call__(self, index, **kwargs):
        cropitem = self.crops[index]
        path = cropitem.path
        if self.cache: 
            # load from memory cache
            tensor = self.cache[path]
        else: 
            # cache is empty dict
            # load from disk
            tensor = cropitem.read()

        cropped = self.crops[index].crop(tensor).unsqueeze(0)

        if self.denan:
            cropped = cropped.nan_to_num(nan=-1.0)
        
        return cropped

    def __getitem__(self, index):
        return self.__call__(index)
    
    def __len__(self):
        return len(self.crops)



def denan(item):
    return item.nan_to_num(nan=-1.0)


def read_volume(path:Path|str) -> torch.Tensor:
    path = Path(path)
    if path.is_dir():
        return read_sequence(path)

    if path.suffix.lower() == ".nrrd":
        # assumes single float, beta scale
        nrrdfile = nrrd.read(str(path))
        result = torch.as_tensor(nrrdfile[0])
        if nrrdfile[1]['type'] != 'float16':
            # rescale from beta scale to 16 bit
            result = Rescale().encodes(result).half()
        return result
    
    try:
        result = torch.load(path)
    except Exception:
        # If it is not a torch tensor, then read using skimage
        result = torch.as_tensor(io.imread(path))

    return result

def iter_slice_paths(directory: Path) -> Iterable[Path]:
    """
    Iterates over the paths of .tif images in a directory, sorted based on the numerical part of their names.

    The function first identifies the most common prefix among the .tif images in the directory. It then sorts the paths
    of the images with this prefix based on the numerical part of their names (assumed to be at the end of the name,
    before the extension). The sorted paths are then returned one by one.

    Args:
        directory (Path): The directory containing the .tif images.

    Yields:
        Iterable[Path]: An iterator over the sorted paths of the .tif images.
    """
    prefixes = [
        re.match(r"(.*?)[0-9]+.tif", path.name).group(1) or ""  # type: ignore
        for path in directory.glob("*[0-9].tif")
    ]
    target_prefix = Counter(prefixes).most_common(1)[0][0]
    paths = sorted(
        directory.glob(f"{target_prefix}*.tif"),
        key=lambda p: int(re.search(r"[0-9]+$", p.stem)[0]),  # type: ignore
    )
    yield from paths

def read_sequence(directory: Path) -> np.ndarray:
    """
    Reads a sequence of .tif images from a directory and returns them as a 3D numpy array.

    Args:
        directory (Path): The directory containing the .tif images.

    Returns:
        npt.NDArray[np.float64]: A 3D numpy array containing the image data. The dimensions are
        (width, height, number of images).
    """
    paths = list(iter_slice_paths(directory))
    size = np.array(Image.open(paths[0]).size)
    seq = np.zeros((*size[::-1], len(paths)), dtype=np.float64)
    for path, ii in tqdm(zip(paths, range(len(paths))), total=len(paths)):
        seq[:, :, ii] = np.array(Image.open(path))
    return seq


def write_volume(data:torch.Tensor, path:Path|str) -> None:
    """
    Saves a 3D volume data to the specified path in either .pt or image format.

    Args:
        data (torch.Tensor): The 3D tensor data to be saved.
        path (Path | str): The destination file path where the data will be saved. If the suffix is
            '.pt', the data will be saved as a PyTorch tensor. Otherwise, it will be saved as an image file.

    Returns:
        None

    Notes:
        - If the directory specified in the path does not exist, it will be created.
        - Supports saving in '.pt' format for PyTorch tensors. Other formats will
          be saved using `io.imsave` from skimage.io after converting the tensor to a NumPy array.
    """
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    match path.suffix.lower():
        case ".nrrd":
            nrrd.write(str(path), data.numpy())
        case ".pt":
            torch.save(data, path)
        case '.tif':
            # no suffix, assumes list of tiff output
            write_sequence(data, path.parent, file_prefix=path.stem)
        case _:
            io.imsave(path, data.numpy())    

def write_sequence(data:torch.tensor, path: Path, axis:int=-1, flip=(0,1), file_prefix:str='out') -> None:
    """
    Writes 3D torch tensor to a sequence of .tif images in a specified directory.
    """
    
    # manage path
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    data = data.squeeze().cpu().numpy()

    if flip is not None:
        data = np.flip(data, axis=flip)
    # number of images
    nslices=data.shape[axis]
    ndigits=len(str(nslices))

    for i in track(range(nslices), description='writing tif slices:'):    
        slice = np.take(data, i, axis=axis)

        # output
        output_filename = path/f"{file_prefix}{i:0{ndigits}d}.tif"
        tifffile.imwrite(output_filename, slice, photometric='minisblack')
    
    print(f'output written to {path}')


class beta_to_12bit(DisplayedTransform):
    """
    Assumes input data in beta scale.
    Standard normalization for 16-bit TIFF output in BCT: (min-in max-in max-out) = (5.0e-11 7.0e-10 4095)
    Image data type will be int16.
    Scalar range will be 12 bit.
    """
    def __init__(self) -> None:
        self.in_min = 5.0e-11
        self.in_max = 7.0e-10
        self.out_max = 4095

    def encodes(self, image: torch.Tensor) -> torch.Tensor:
        image_dtype = image.dtype
        image = image.double()
        image = (image - self.in_min) / (self.in_max - self.in_min) * self.out_max
        # uint16 not supported by pytorch
        image = torch.clamp(image, 0, self.out_max).type(torch.int16)
        return image.type(image_dtype)
    

def thick_axial(input:torch.Tensor, axis:int=0, thickness:int=30, spacing:int=15, threshold:float=2e-10, MIP:bool=False) -> torch.Tensor:
    '''
    Given input node, compute thick slices and return output node.
    plane is in {0, 1, 2} corresponding to array data dimensions.
    thickness and spacing are multiplying factors of the original spacing.
    '''

    # get array
    source_array = input.squeeze().numpy().copy()

    # move chosen plane to first dimension
    source_array = np.moveaxis(source_array, axis, 0)

    # get original dimensions and compute new dimensions
    nThin = source_array.shape[0]
    nThick = 1 + int((nThin - thickness) / spacing)
    output_array = np.zeros([nThick] + list(source_array.shape[1:]), dtype=source_array.dtype)

    # identify high voxels
    if threshold is not None:
        high = source_array > threshold
        # count neighbours, we exclude isolated upper voxels as noise
        kernel = np.ones((3, 3, 3), dtype=int)
        kernel[1, 1, 1] = 0  

        # convolve to count neighbours
        n_neighbours = ndimage.convolve(high.astype(np.uint8), kernel)

        # remove isolated voxels
        high[n_neighbours<=1] = 0

    # init loop
    for i in track(range(nThick),total=nThick,description='Thick slicing: '):
        i_min=i*spacing
        i_max=i_min+thickness
        # compute thick slice by mean
        thins = source_array[i_min:i_max, ...]
        if MIP:
            # Maximum Intensity Projection
            thick = np.max(thins, axis=0)
        else:
            # Use average
            thick = np.mean(thins, axis=0)

        if threshold is not None:
            # get high voxels 
            thins_upper = high[i_min:i_max, ...]
            # identify voxels above threshold in thick
            thick_upper = np.any(thins_upper, axis=0)
            # get mean of voxels above threshold
            thins_hi = thins.copy()
            thins_hi[~thins_upper] = np.nan
            thick_hi = np.nanmean(thins_hi, axis=0)
            # set voxels above threshold to mean above threshold
            thick[thick_upper] = thick_hi[thick_upper]
        output_array[i, ...] = thick
    # move back to original dimensions, conver to tensor
    output_array = torch.Tensor(np.moveaxis(output_array, 0, axis))

    return output_array

def write_dicom(data:torch.Tensor, path:Path|str, file_prefix:str='out', axis:int=0, dcm_name='name', dcm_sex='F', dcm_laterality='R', dcm_details='details') -> None:
    '''
    path should be a directory
    '''

    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)


    # create DICOM template
    ds = pydicom.dataset.Dataset()
    ds.PatientID = dcm_name # VAR
    ds.PatientName = dcm_name # VAR
    ds.PatientSex = dcm_sex # VAR
    ds.ImageLaterality = dcm_laterality # FIX
    # ds.AccessionNumber = dcm_details # VAR
    ds.AccessionNumber = pydicom.uid.generate_uid()
    # ds.SeriesInstanceUID = dcm_details # VAR
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesDescription = dcm_details # VAR

    # fixed tags
    ds.PatientBirthDate = "19800101" # FIX
    # ds.StudyInstanceUID = "StudyID" # FIX
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.StudyID = dcm_details
    ds.StudyDescription = dcm_details


    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
    ds.Modality = "MG" # FIX
    ds.ViewPosition = "CC" # FIX
    # ds.NominalScannedPixelSpacing = [0.1, 0.1] 
    ds.PixelSpacing = [0.1, 0.1]
    ds.SliceThickness = 3.0
    ds.SpacingBetweenSlices = 1.5
    ds.WindowWidth = "1150" # FIX
    ds.WindowCenter = "625" # FIX
    ds.ConversionType = "WSD" # FIX
    ds.StudyDate = datetime.now().strftime('%Y%m%d')
    ds.SeriesNumber = None

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15        
    ds.PixelRepresentation = 0 # 1 means signed integers (as np.int16 is signed)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.RescaleType = "US"

    ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage

    # create required file meta info
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    file_meta.FileMetaInformationGroupLength = 238
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.ImplementationVersionName = f'pydicom {pydicom.__version__}'
    file_meta.SourceApplicationEntityTitle = 'pydicom'
    ds.file_meta = file_meta
    ds.preamble=b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

    # prep data
    # numpy supports uint16
    data = data.squeeze().cpu().numpy().astype(np.uint16)
    data = data.view(data.dtype.newbyteorder('<'))
    data = np.flip(data, axis=(1,2))

    nslices=data.shape[axis]
    ndigits=len(str(nslices))
    # number of images
    for i in range(nslices):
        dsc=ds.copy()
        dsc.SOPInstanceUID = pydicom.uid.generate_uid()
        now = datetime.now()
        dsc.StudyTime = str(float(now.strftime('%H%M%S'))+(now.microsecond/1_000_000))

        slice = np.take(data, indices=i, axis=axis).T
        # set pixel data
        dsc.Rows = slice.shape[0]
        dsc.Columns = slice.shape[1]

        dsc.PixelData = slice.tobytes()

        # spatial 
        dsc.InstanceNumber = i + 1 # 1 for first image
        dsc.ImagePositionPatient = [0.0, 0.0, i*ds.SpacingBetweenSlices]
        dsc.SliceLocation = i*ds.SpacingBetweenSlices

        # output
        output_filename = path/f"{file_prefix}{i+1:0{ndigits}d}.dcm"
        pydicom.filewriter.dcmwrite(output_filename, dsc)



def unsqueeze(inputs):
    """Adds a dimension for the single channel."""
    return inputs.unsqueeze(dim=1)


def crop(x):
    center_i = x.shape[0] // 2
    center_j = x.shape[1] // 2
    center_k = x.shape[2] // 2

    size=32

    return x[center_i-size:center_i+size,center_j-size:center_j+size,center_k-size:center_k+size] # hack


def VolumeBlock(*args, **kwargs):
    type_tfms = [read_volume]
    if 'type_tfms' in kwargs:
        type_tfms.append(kwargs.pop('type_tfms'))

    return TransformBlock(
        *args,
        type_tfms=type_tfms,
        batch_tfms=unsqueeze,
        **kwargs,
    )


class CropTransform(DisplayedTransform):
    def __init__(self, start_x:int=None, end_x:int=None, start_y:int=None, end_y:int=None, start_z:int=None, end_z:int=None ):
        self.start_x = start_x or None
        self.end_x = end_x or None
        self.start_y = start_y or None
        self.end_y = end_y or None
        self.start_z = start_z or None
        self.end_z = end_z or None

    def encodes(self, data):
        if isinstance(data, PILImageBW):
            data = np.expand_dims(np.asarray(data), 0)

        if len(data.shape) == 3:
            return data[self.start_z:self.end_z,self.start_y:self.end_y,self.start_x:self.end_x]
        return data[self.start_y:self.end_y,self.start_x:self.end_x]        

    
class Rescale(DisplayedTransform):
    '''
    Rescales the image between rescale_min and rescale_max with rescale_factor.
    The defaults are from chosen reconstruction limits that correspond to the 
    beta value range seen in breast tissue.
    '''
    def __init__(self, rescale_min=5.0e-11, rescale_max=7.0e-10, rescale_factor=10.0) -> None:
        self.rescale_min = rescale_min
        self.rescale_max = rescale_max
        self.rescale_factor = rescale_factor

    def encodes(self, image: torch.Tensor) -> torch.Tensor:
        image_dtype = image.dtype
        image = image.double()
        image = (image - self.rescale_min) / (self.rescale_max - self.rescale_min) * self.rescale_factor
        return image.type(image_dtype)

    def decodes(self, image: torch.Tensor) -> torch.Tensor:
        image_dtype = image.dtype
        if not image_dtype == torch.float32 and not image_dtype==torch.float64:
            raise TypeError(f"Input must be float32 or float64 to avoid precision loss. Got {image_dtype}")
        image = image.double()
        image = (image / self.rescale_factor) * (self.rescale_max - self.rescale_min) + self.rescale_min
        return image.type(image_dtype)


class KeepCalcifications(DisplayedTransform):
    '''
    Keeps calcification voxels from input noisy image.
    Calcifications are defined as voxels with beta values > 2e-10
    '''
    def __init__(self, manual_threshold=None, beta_format=None) -> None:
        '''
        manual_threshold: float or None
            If None, threshold is inferred from the input image
            If float, the threshold is set to that value
        beta_format: bool or None 
            ignored if manual_threshold is not None
            If True, the threshold is set to 2e-10
            If False, the threshold is set to the equivalent of 2e-10
        '''
        default_threshold = torch.tensor(2e-10)
        if manual_threshold is not None:
            self.threshold = manual_threshold
        else:
            if beta_format is True:
                self.threshold = default_threshold.item()
            elif beta_format is False:
                # rescaled equivalent
                self.threshold = Rescale().encodes(default_threshold).item()
            elif beta_format is None: 
                raise ValueError("manual_threshold or beta_format must be set")

    def encodes(self, image: torch.Tensor, noisy_image:torch.Tensor) -> torch.Tensor:
        assert image.shape == noisy_image.shape, "Input shapes must match"
        if self.threshold < 1e-5:
            if image.dtype == torch.float16 or noisy_image.dtype == torch.float16:
                raise TypeError("Threshold too small to support float16. May need to set beta_format to False")
        calcification_mask = noisy_image >= self.threshold
        image[calcification_mask] = noisy_image[calcification_mask]

        return image
