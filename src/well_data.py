import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import toml
import zipfile

from os import makedirs, walk as dir_ls
from os.path import isdir, basename, join as join_paths
from time import time
from typing import Dict

import logging
logging.basicConfig(format='[%(asctime)s.%(msecs)03d] [well_data] [%(levelname)s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger()


"""
    Extracts signals from multi-well microscope data and stores each wells signal data as a separate xlsx file.
    Creates an image with all the roi's drawn on one frame & creates an image with signal plots for each well.
"""


def well_data(setup_config: Dict):
    """
        Extracts time series data for defined roi's within each frame of a video and
        writes the results to xlsx files for each roi and zips them into a single archive.
        Also creates an image with all the roi's drawn on one frame as a quick sanity check,
        and creates another image with time series plots of the signal data for each well.
    """

    # read image parameters and declare variables
    num_horizontal_pixels = int(setup_config['num_horizontal_pixels'])
    num_vertical_pixels = int(setup_config['num_vertical_pixels'])
    num_frames = int(setup_config['num_frames'])
    bit_depth = int(setup_config['bit_depth'])
    numWellsH = setup_config['stage']['numWellsH']
    numWellsV = setup_config['stage']['numWellsV']
    nFramesH = setup_config['cols']
    nFramesV = setup_config['rows']
    setup_config['num_well_rows'] = nFramesV * numWellsV
    setup_config['num_well_cols'] = nFramesH * numWellsH
    num_wells = setup_config['stage']['num_wells']
    signal_values = np.empty((num_wells, num_frames), dtype=np.float32)

    #generate ROIs from settings.toml parameters
    x_starts, x_stops, y_starts, y_stops = make_rois(setup_config)

    
    if(bit_depth == 8):
        pixel_np_data_type = np.uint8
        pixel_size = 1
    elif (bit_depth == 12):
        pixel_np_data_type = np.dtype('<u2')
        pixel_size = 2
        log.info(pixel_np_data_type)
    elif (bit_depth == 16):
        pixel_np_data_type = np.dtype('<u2')
        pixel_size = 2
        log.info(pixel_np_data_type)
    else:
        #THIS CONDITION SHOULD THROW AN ERROR RATHER THAN JUST PRINT TO STANDARD OUTPUT
        log.error(f"{bit_depth} bit images are not supported")
        return(1)

    #TODO: Check to see if the file size is correct based on the values of num_horizontal_pixels, num_vertical_pixels, num_frames, pixel_size, i.e. the file size should be num_horizontal_pixels*num_vertical_pixels*num_frames*pixel_size bytes

    # safely create the output dir if it does not exist
    log.info("Creating Output Dir")
    if not isdir(setup_config['output_dir_path']):
        makedirs(name=setup_config['output_dir_path'], exist_ok=False)

    
    # save an image with the roi's drawn on it as quick sanity check.
    log.info("Creating ROI Sanity Check Image...")
    frame_to_draw_rois_on = np.fromfile(file=setup_config['input_path'],dtype=pixel_np_data_type,count=(num_horizontal_pixels*num_vertical_pixels ))
    frame_to_draw_rois_on  = frame_to_draw_rois_on.reshape(num_vertical_pixels ,num_horizontal_pixels)
    path_to_save_frame_image = join_paths(setup_config['output_dir_path'], 'roi_locations.png')
    if (pixel_size == 2):
        frame_to_draw_rois_on = frame_to_draw_rois_on/(frame_to_draw_rois_on.max())
        frame_to_draw_rois_on = frame_to_draw_rois_on * 255
        frame_to_draw_rois_on = frame_to_draw_rois_on.astype('uint8')
    frame_with_rois_drawn(frame_to_draw_rois_on, x_starts, x_stops, y_starts, y_stops, path_to_save_frame_image)
    log.info("ROI Sanity Check Image Created")

    # extract ca2+ signal in each well for each frame
    """
    Extracts signals from multi-well microscope data and stores each wells signal data as a separate xlsx file.
    Creates an image with all the roi's drawn on one frame & creates an image with signal plots for each well.
    """
    log.info("Starting Signal Extraction...")
    StartTime = time()
    frame_num=0
    while (frame_num < num_frames):
        i=0
        currentFrame = np.fromfile(file=setup_config['input_path'],dtype=pixel_np_data_type ,count=int(num_horizontal_pixels*num_vertical_pixels ),offset = int(frame_num*num_horizontal_pixels*num_vertical_pixels*pixel_size))
        currentFrame = currentFrame.reshape(num_vertical_pixels ,num_horizontal_pixels)
        while (i < num_wells):
            x_start = x_starts[i]
            x_end = x_stops[i]
            y_start = y_starts[i]
            y_end = y_stops[i]
            signal_values[i, frame_num] = np.mean(currentFrame[y_start:y_end, x_start:x_end])
            i=i+1
        frame_num = frame_num + 1
    log.info("Signal Extraction Complete")
    StopTime = time()
    log.info(f"Processed signals in {(StopTime - StartTime)} seconds")

    # write each roi's time series data to an xlsx file
    if setup_config['stage']['save_excel']:
        save_excel_files(signal_values = signal_values, setup_config=setup_config)
        
    log.info("Writing ROI Signals to CSV file...")
    StartTime = time()
    csvFilePath = join_paths(setup_config['output_dir_path'], 'results.csv')
    signal_values_t = signal_values.transpose()
    np.savetxt(csvFilePath , signal_values_t, delimiter=',')
    StopTime = time()
    log.info(f"CSV created in {(StopTime - StartTime)} seconds")
    
    if setup_config['save_plots']:
        signals_to_plot(signal_values, setup_config)
     






"""

Handles some logistics for saving excel files

"""
def save_excel_files(signal_values: np.ndarray, setup_config: Dict):
    log.info("Writing ROI Signals to XLSX files...")
    StartTime = time()
    time_stamps = np.linspace(start=0, stop=setup_config['duration'], num=setup_config['num_frames'])
    setup_config['xlsx_output_dir_path'] = join_paths(setup_config['output_dir_path'], 'xlsx')
    if not isdir(setup_config['xlsx_output_dir_path']):
        makedirs(name=setup_config['xlsx_output_dir_path'], exist_ok=False)
    #make_xlsx_output_dir(xlsx_output_dir_path=setup_config['xlsx_output_dir_path'])
    signal_to_xlsx_for_sdk(signal_values = signal_values, time_stamps = time_stamps, setup_config = setup_config)
    log.info("Writing Signals to XLSX Files Complete")
    StopTime = time()
    log.info(f"XLSX files created in {(StopTime - StartTime)} seconds")
    # zip all the xlsx files into a single archive
    log.info("Creating Zip Archive For XLSX files...")
    StartTime = time()
    xlsx_archive_file_path = join_paths(setup_config['output_dir_path'], 'xlsx-results.zip')
    zip_files(input_dir_path=setup_config['xlsx_output_dir_path'], zip_file_path=xlsx_archive_file_path)
    log.info("Zip Archive For XLSX files Created")
    StopTime = time()
    log.info(f"XLSX zipped in {(StopTime - StartTime)} seconds")

"""

Graphs time series data for each well

"""

def make_rois(setup_config: Dict):
    frameCentersH = np.linspace(0.5, setup_config['cols']-0.5, setup_config['cols'])*setup_config['width'] + setup_config['stage']['hOffset']
    frameCentersV = np.linspace(0.5, setup_config['rows']-0.5, setup_config['rows'])*setup_config['height'] - setup_config['stage']['vOffset']
    nFramesH = setup_config['cols']
    nFramesV = setup_config['rows']
    numWellsH = setup_config['stage']['numWellsH']
    numWellsV = setup_config['stage']['numWellsV']
    num_wells = setup_config['stage']['num_wells']
    pixelSize = setup_config['xy_pixel_size']*setup_config['scale_factor']
    #roiSize = setup_config['stage']['roiSize']
    roiSizeX = setup_config['stage']['roiSizeX']
    roiSizeY = setup_config['stage']['roiSizeY']
    RowNames = ["A","B","C","D","E","F","G","H", "I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X", "Y","Z","AA","AB","AC","AD","AE","AF"]
    wellSpacing = setup_config['stage']['wellSpacing']
    x_starts = np.empty(num_wells, dtype=np.int64)
    y_starts = np.empty(num_wells, dtype=np.int64)
    x_stops = np.empty(num_wells, dtype=np.int64)
    y_stops = np.empty(num_wells, dtype=np.int64)
    wellNames = list()
    wellRows = list()
    wellColumns = list()
    n=0
    l=0
    while(l < nFramesV):
        frameCenterV = frameCentersV[l]
        j = 0
        while(j < numWellsV):
            k = 0
            while(k < nFramesH):
                frameCenterH = frameCentersH[k]
                i = 0
                while(i < numWellsH):
                    #x_starts[n]=frameCenterH-roiSize/2 + (wellSpacing/(2*pixelSize)) - (numWellsH/2 - i) * (wellSpacing/pixelSize)
                    x_starts[n]=frameCenterH-roiSizeX/2 + (wellSpacing/(2*pixelSize)) - (numWellsH/2 - i) * (wellSpacing/pixelSize)
                    #x_stops[n] = x_starts[n] + roiSize
                    x_stops[n] = x_starts[n] + roiSizeX
                    #y_starts[n] = frameCenterV-roiSize/2 + (wellSpacing/(2*pixelSize)) - (numWellsV/2 - j) * (wellSpacing/pixelSize)
                    y_starts[n] = frameCenterV-roiSizeY/2 + (wellSpacing/(2*pixelSize)) - (numWellsV/2 - j) * (wellSpacing/pixelSize)
                    #y_stops[n] = y_starts[n] + roiSize
                    y_stops[n] = y_starts[n] + roiSizeY
                    #print(f"{RowNames[numWellsV*l+j]}{(numWellsH*k+i+1)}")
                    wellNames.append(f"{RowNames[numWellsV*l+j]}{(numWellsH*k+i+1)}")
                    wellRows.append(numWellsV*l+j)
                    wellColumns.append(numWellsH*k+i+1)
                    n = n + 1
                    i = i+1
                k = k + 1
            j = j +1
        l=l+1
    setup_config['x_starts'] = x_starts
    setup_config['x_stops'] = x_stops
    setup_config['y_starts'] = y_starts
    setup_config['y_stops'] = y_stops
    setup_config['wellNames'] = wellNames
    setup_config['wellRows'] = wellRows
    setup_config['wellColumns'] = wellColumns
    return x_starts, x_stops, y_starts, y_stops

def signals_to_plot(signal_values: np.ndarray, setup_config: Dict):
    """ Create an image with plots of time series data for multiple ROIs """
    StartTime = time()
    log.info("Creating Signal Plot Sanity Check Image...")
    time_stamps = np.linspace(start=0, stop=setup_config['duration'], num=setup_config['num_frames'])
    num_wells, num_data_points = signal_values.shape
    #setup_config['num_well_rows'] = nFramesV * numWellsV
    #setup_config['num_well_cols'] = nFramesH * numWellsH
    plot_file_path = join_paths(setup_config['output_dir_path'], 'roi_signals_plots.' + setup_config['plot_format'])

    fig, axes = plt.subplots(
        nrows=setup_config['num_well_rows'], ncols=setup_config['num_well_cols'],
        figsize=(setup_config['num_well_cols']*3, setup_config['num_well_rows']*3),
        dpi=300.0,
        layout='constrained'
    )

    instrument_name = setup_config['instrument_name']
    recording_date = setup_config['recording_date']
    video_file_name = basename(setup_config['input_path'])
    plot_title = f"{instrument_name} Experiment Data - {recording_date} - {video_file_name}"

    fig.suptitle(plot_title, fontsize=20)
    fig.supylabel('ROI Average')
    fig.supxlabel('Time (s)')

    
    i = 0
    while (i < num_wells):
        serial_position = i
        well_signal = signal_values[serial_position, :]
        plot_row = setup_config['wellRows'][i]
        plot_col = setup_config['wellColumns'][i] - 1
        axes[plot_row, plot_col].plot(time_stamps, well_signal)
        axes[plot_row, plot_col].set_title(setup_config['wellNames'][i])
        i = i + 1

    plt.savefig(plot_file_path)
    log.info("Signal Plot Sanity Check Image Created")
    StopTime = time()
    log.info(f"Sanity check ran in {(StopTime - StartTime)} seconds")

"""

zips input directory

"""
def zip_files(input_dir_path: str, zip_file_path: str):
    zip_file = zipfile.ZipFile(zip_file_path, 'w')
    for dir_name, _, file_names in dir_ls(input_dir_path):
        for file_name in file_names:
            file_path = join_paths(dir_name, file_name)
            zip_file.write(file_path, basename(file_path))
    zip_file.close()

"""

saves time series data as excel files

"""
def signal_to_xlsx_for_sdk(signal_values: np.ndarray, time_stamps: np.ndarray, setup_config: Dict):
    """ writes time series data to xlsx files for multiple ROIs """

    num_wells, num_data_points = signal_values.shape
    frames_per_second = setup_config['fps']
    date_stamp = setup_config['recording_date']
    output_dir = setup_config['xlsx_output_dir_path']
    data_type = setup_config['data_type']

    if 'barcode' in setup_config:
        well_plate_barcode = setup_config['barcode']
    else:
        well_plate_barcode = 'NA'
    i = 0
    while (i < num_wells):
        #print(i)
        well_name = setup_config['wellNames'][i]
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet['E2'] = well_name 
        sheet['E3'] = date_stamp
        sheet['E4'] = well_plate_barcode
        sheet['E5'] = frames_per_second
        sheet['E6'] = 'y'  # do twitch's point up
        sheet['E7'] = 'NAUTILUS'  # microscope name
        sheet['E9'] = data_type # voltage or calcium imaging
        template_start_row = 2
        time_column = 'A'
        signal_column = 'B'
        well_data_row = i
        for data_point_position in range(num_data_points):
            sheet_row = str(data_point_position + template_start_row)
            sheet[time_column + sheet_row] = time_stamps[data_point_position]
            sheet[signal_column + sheet_row] = signal_values[well_data_row, data_point_position]
        path_to_output_file = join_paths(output_dir, well_name + '.xlsx')
        workbook.save(filename=path_to_output_file)
        workbook.close()
        i = i + 1

def frame_with_rois_drawn(frame_to_draw_on: np.ndarray,x_starts: np.ndarray,x_stops: np.ndarray,y_starts: np.ndarray,y_stops: np.ndarray, path_to_save_frame_image: str):
  """ Draw multiple ROIs on one frame image """
  i = 0
  green_line_colour_bgr = (0, 255, 0)
  #frame_to_draw_on = asUINT8(frame_to_draw_on)
  frame_min = np.min(frame_to_draw_on)
  frame_max = np.max(frame_to_draw_on)
  frame_range = frame_max - frame_min
  frame_to_draw_on = np.uint8(255.0*(frame_to_draw_on - frame_min)/frame_range)
  frame_to_draw_on = np.stack((frame_to_draw_on, frame_to_draw_on, frame_to_draw_on), axis=-1)
  while(i < x_starts.size):  
      top_left = (
          int(x_starts[i]),
          int(y_starts[i])
      )
      lower_right = (
          int(x_stops[i]),
          int(y_stops[i])
          
      )
      cv.rectangle(
          img=frame_to_draw_on,
          pt1=top_left,
          pt2=lower_right,
          color=green_line_colour_bgr,
          thickness=1,
          lineType=cv.LINE_AA
      )
      i = i + 1
  cv.imwrite(path_to_save_frame_image, frame_to_draw_on)
  
#def make_output_dir(output_dir_path: str):
#    """ create the main output dir """
#    if not isdir(output_dir_path):
#        makedirs(name=output_dir_path, exist_ok=False)

def main():
    
    parser = argparse.ArgumentParser(description='Extracts signals from a multi-well microscope experiment')
    parser.add_argument(
        'toml_config_path',
        default=None,
        help='Path to a toml file with run config parameters'
    )
    parser.add_argument(
        '--input_video_path',
        default=None,
        help='Path to a video with multi-well data',
    )
    parser.add_argument(
        '--output_dir_path',
        default=None,
        help='Path to save all output',
    )
    parser.add_argument(
        '--num_horizontal_pixels',
        default=None,
        help='Number of horizontal pixels',
    )
    parser.add_argument(
        '--num_vertical_pixels',
        default=None,
        help='Number of vertical pixels',
    )
    parser.add_argument(
        '--num_frames',
        default=None,
        help='Number of frames',
    )
    parser.add_argument(
        '--bit_depth',
        default=None,
        help='number of bits per pixel',
    )
    parser.add_argument(
        '--scale_factor',
        default=None,
        help='Scaling factor, a 3072x2048 image has a scale factor of 1, a 1536x1024 has a scale factor of 2',
    )
    parser.add_argument(
        '--duration',
        default=None,
        help='Duration of recording, in seconds',
    )
    parser.add_argument(
        '--fps',
        default=None,
        help='number of frames per second',
    )
    args = parser.parse_args()

    
    #print(args.toml_config_path)
    toml_file = open(args.toml_config_path)
    setup_config = toml.load(toml_file)

    

    if args.input_video_path is not None:
        setup_config['input_path'] = args.input_video_path
    if args.output_dir_path is not None:
        setup_config['output_dir_path'] = args.output_dir_path
    if args.num_horizontal_pixels is not None:
        setup_config['num_horizontal_pixels'] = args.num_horizontal_pixels
    if args.num_vertical_pixels is not None:
        setup_config['num_vertical_pixels'] = args.num_vertical_pixels
    if args.num_frames is not None:
        setup_config['num_frames'] = args.num_frames
    if args.bit_depth is not None:
        setup_config['bit_depth'] = args.bit_depth
    if args.scale_factor is not None:
        setup_config['scale_factor'] = int(args.scale_factor)
    if args.duration is not None:
        setup_config['duration'] = float(args.duration)
    if args.fps is not None:
        setup_config['fps'] = float(args.fps)
    
    if 'save_excel' not in setup_config['stage']:
        setup_config['stage']['save_excel'] = True
    if 'save_plots' not in setup_config:
        setup_config['save_plots'] = True
    if 'plot_format' not in setup_config:
        setup_config['plot_format'] = 'png'
    if 'fluorescence_normalization' not in setup_config:
        setup_config['fluorescence_normalization'] = 'None'
        
    
    well_data(setup_config=setup_config)

    toml_file.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.error(f"Unhandled exception {str(e)}")
