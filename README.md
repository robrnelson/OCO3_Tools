# OCO3_plot_SAM.py
A simple Python tool to load data from an OCO-3 .nc4 Lite file and plot the quality flagged, bias corrected XCO2 for a single Snapshot Area Mapping (SAM) mode observation on a map.  

## Data

OCO-3 v11 .nc4 Lite files can be download from GESDISC:

<https://disc.gsfc.nasa.gov/datasets/OCO3_L2_Lite_FP_11r/summary?keywords=oco3>

There are mulitple ways to download the data shown on the right under "Data Access"

## Usage
python OCO3_plot_SAM.py filename site

## Example
python OCO3_plot_SAM.py oco3_LtCO2_211218_B11072Ar_240915205334s.nc4 fossil_Los_Angeles_USA

