#!/usr/bin/env python3
import h5py
import numpy as np
import argparse
import scipy.cluster.hierarchy as hcluster
import calendar
import matplotlib.pyplot as plt
from pyproj import Proj, transform
from PIL import Image
import urllib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes

def load_data(filename):

    """
    Load data from an OCO-3 .nc4 Lite file.
    """

    # Open file
    data_file = h5py.File(filename, "r")

    nsoundings = len(data_file['sounding_id'])

    data_temp = np.zeros((nsoundings),dtype=[\
        ('sounding_id','int'),
        ('latitude','f8'),
        ('longitude','f8'),
        ('time','f8'),
        ('vertex_latitude_1','f8'),
        ('vertex_latitude_2','f8'),
        ('vertex_latitude_3','f8'),
        ('vertex_latitude_4','f8'),
        ('vertex_longitude_1','f8'),
        ('vertex_longitude_2','f8'),
        ('vertex_longitude_3','f8'),
        ('vertex_longitude_4','f8'),
        ('sounding_operation_mode','S2'),
        ('sounding_operation_mode_string','S6'),
        ('production_string','S100'),
        ('target_id','S100'),
        ('target_name','S100'),
        ('target_lat','f8'),
        ('target_lon','f8'),
        ('xco2_quality_flag','f8'),
        ('orbit','f8'),
        ('xco2_bc','f8')])

    data_temp['sounding_id'] = np.array(data_file['sounding_id'][...])
    data_temp['time'] = np.array(data_file['time'][...])
    data_temp['latitude'] = np.array(data_file['latitude'][...])
    data_temp['longitude'] = np.array(data_file['longitude'][...])

    vertex_latitude_temp = np.array(data_file['vertex_latitude'][...])
    vertex_longitude_temp = np.array(data_file['vertex_longitude'][...])
    data_temp['vertex_latitude_1'] = vertex_latitude_temp[:,0]
    data_temp['vertex_latitude_2'] = vertex_latitude_temp[:,1]
    data_temp['vertex_latitude_3'] = vertex_latitude_temp[:,2]
    data_temp['vertex_latitude_4'] = vertex_latitude_temp[:,3]
    data_temp['vertex_longitude_1'] = vertex_longitude_temp[:,0]
    data_temp['vertex_longitude_2'] = vertex_longitude_temp[:,1]
    data_temp['vertex_longitude_3'] = vertex_longitude_temp[:,2]
    data_temp['vertex_longitude_4'] = vertex_longitude_temp[:,3]

    #Convert integers to strings to be more useful
    data_temp['sounding_operation_mode'][np.array(data_file['Sounding/operation_mode'][...]) == 0] = b'ND'
    data_temp['sounding_operation_mode'][np.array(data_file['Sounding/operation_mode'][...]) == 1] = b'GL'
    data_temp['sounding_operation_mode'][np.array(data_file['Sounding/operation_mode'][...]) == 2] = b'TG'
    data_temp['sounding_operation_mode'][np.array(data_file['Sounding/operation_mode'][...]) == 3] = b'XS'
    data_temp['sounding_operation_mode'][np.array(data_file['Sounding/operation_mode'][...]) == 4] = b'AM'

    #Strings for plotting titles
    data_temp['sounding_operation_mode_string'][data_temp['sounding_operation_mode'] == b'TG'] = "Target"
    data_temp['sounding_operation_mode_string'][data_temp['sounding_operation_mode'] == b'AM'] = "SAM"
    data_temp['sounding_operation_mode_string'][data_temp['sounding_operation_mode'] == b'ND'] = "Nadir"
    data_temp['sounding_operation_mode_string'][data_temp['sounding_operation_mode'] == b'GL'] = "Glint"

    #Load the bias corrected XCO2 and corresponding XCO2 quality flag
    data_temp['xco2_bc'] = np.array(data_file['xco2'][...])
    data_temp['xco2_quality_flag'] = np.array(data_file['xco2_quality_flag'][...])

    data_temp['target_id'] = np.array(data_file['Sounding/target_id'][...])
    data_temp['target_name'] = np.array(data_file['Sounding/target_name'][...])
    data_temp['orbit'] = np.array(data_file['/Sounding/orbit'][...])
    data_temp['production_string'] = data_file.attrs['CollectionLabel']

    data_file.close()

    #Remove bad quality retrievals
    mask_temp = data_temp['xco2_quality_flag'] == 0
    data_temp = data_temp[mask_temp]

    #Remove bad lats (rare)
    data_temp = data_temp[(data_temp['latitude'] > -999999.)&(data_temp['latitude'] != 0.0)]

    #Remove bad vertex coordinates (rare)
    data_temp = data_temp[(data_temp['vertex_latitude_1'] > -999999.)&(data_temp['vertex_latitude_2'] > -999999.0)&(data_temp['vertex_latitude_3'] > -999999.0)&(data_temp['vertex_latitude_4'] > -999999.0)]

    return data_temp


def plot_data(data_temp, site_temp):

    """
    Plot quality flagged, bias corrected XCO2 on a map.
    """

    #Crop the data to only include the site of interest
    data_temp = data_temp[data_temp["target_name"].astype(str) == site_temp]
   
    #Plot the data
    var_temp = 'xco2_bc'
    var_string = "xco2_bc_qf"
    var_cbar_string = r'$X_{CO_2}$'+' [ppm]'
    var_title_string = 'Bias Corrected and Quality Flagged '+ r'$X_{CO_2}$'
    var_cbar = plt.cm.viridis

    #Remove weird outliers that are far from the actual TG/AM. This happens sometimes when the instrument swings far off-nadir when transitioning to the next SAM swath.
    cluster_input = np.array([data_temp['longitude'],data_temp['latitude']]).T
    try:
      clusters = hcluster.fclusterdata(cluster_input,1.,criterion='distance')
      #Grab the largest cluster, or any/all clusters greater than 32 soundings
      cluster_mask = (clusters == np.argmax(np.bincount(clusters))) | np.isin(clusters, np.where(np.bincount(clusters) > 32)[0])

    except:
      #If there's exactly one sounding, fclusterdata doesn't work
      cluster_mask = np.ones(len(data_temp),dtype=bool)

    #Determine plotting box
    N,S,E,W = np.round(data_temp['latitude'][cluster_mask].max()+.25,1),\
              np.round(data_temp['latitude'][cluster_mask].min()-.25,1),\
              np.round(data_temp['longitude'][cluster_mask].max()+(.25/np.cos(np.deg2rad(data_temp['latitude'][cluster_mask].max()))),1),\
              np.round(data_temp['longitude'][cluster_mask].min()-(.25/np.cos(np.deg2rad(data_temp['latitude'][cluster_mask].min()))),1)

    #Keep anything within the plotting box
    data_temp = data_temp[(data_temp['latitude'] < N)&(data_temp['latitude'] > S)&(data_temp['longitude']<E)&(data_temp['longitude']>W)]

    #Rare situation where we view the same site twice on the same day. Could handle this gracefully, but for now just print a warning.
    if (data_temp['time'].max() - data_temp['time'].min())/60. > 5: print("WARNING: plotting multiple SAMs taken on the same day!")

    #Get some datetime info from the sounding_id
    month_str = calendar.month_abbr[int(data_temp['sounding_id'][0].astype(str)[4:6].lstrip("0"))]
    day_str, year_str = data_temp['sounding_id'][0].astype(str)[6:8].lstrip("0"), data_temp['sounding_id'][0].astype(str)[:4].lstrip("0")
    hour_str, minute_str = data_temp['sounding_id'][0].astype(str)[8:10],data_temp['sounding_id'][0].astype(str)[10:12]

    #Plot
    fig = plt.figure(figsize=(20,20))
    ax1 = plt.axes(projection=ccrs.epsg(3857))
    ax1.set_extent([W,E,S,N], ccrs.PlateCarree())
    ax1.coastlines()

    #Grid
    gl = ax1.gridlines(draw_labels=True, color="0.75")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 24}
    gl.ylabel_style = {'size': 24}

    #Background map
    m1 = Proj("epsg:3857", preserve_units=True)
    W_3857, S_3857 = m1(W,S)
    E_3857, N_3857 = m1(E,N)
    xpixels = 2000
    ypixels = int((N_3857 - S_3857) / (E_3857 - W_3857) * xpixels)
    url = f'http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={W_3857},{S_3857},{E_3857},{N_3857}&bboxSR=3857&imageSR=3857&size={xpixels},{ypixels},&dpi=96&format=png32&transparent=true&f=image'
    try: ESRI = np.array(Image.open(urllib.request.urlopen(url)))
    except: #Sometimes this fails randomly, so try again
      time.sleep(20)
      ESRI = np.array(Image.open(urllib.request.urlopen(url)))
    im1 = ax1.imshow(ESRI,extent=ax1.get_extent(),origin="upper")

    #Plot footprints
    patches = []
    for j in range(len(data_temp['vertex_longitude_1'])):
      if (data_temp['vertex_longitude_1'][j] == 0.0)|(data_temp['vertex_longitude_2'][j] == 0.0)|(data_temp['vertex_longitude_1'][j] == 0.0)|(data_temp['vertex_longitude_1'][j] == 0.0): print("Bad vertex...")
      else: patches += [Polygon([(data_temp['vertex_longitude_1'][j],data_temp['vertex_latitude_1'][j]),(data_temp['vertex_longitude_2'][j],data_temp['vertex_latitude_2'][j]),(data_temp['vertex_longitude_3'][j],data_temp['vertex_latitude_3'][j]),(data_temp['vertex_longitude_4'][j],data_temp['vertex_latitude_4'][j])])]
    p = PatchCollection(patches, alpha=1., transform=ccrs.PlateCarree())
    p.set_array(data_temp[var_temp])
    p.set_clim(np.percentile(data_temp[var_temp],10),np.percentile(data_temp[var_temp],90))
    p.set_lw(1.0)
    p.set_cmap(var_cbar)
    ax1.add_collection(p)

    #Colorbar
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size=0.4, pad=0.25, axes_class=plt.Axes)
    if var_temp == 'aod_total': cbar = fig.colorbar(p,extend='max',cax=cax)
    else: cbar = fig.colorbar(p,extend='both',cax=cax)
    cbar.set_label(var_cbar_string,size=28,rotation=270,labelpad=35)
    cbar.ax.tick_params(labelsize=22)
    cbar.ax.yaxis.get_offset_text().set_fontsize(22)

    #Title
    title = ax1.set_title('OCO-3 '+var_title_string+'\n'+data_temp['sounding_operation_mode_string'][0].astype(str)+' Mode, '+data_temp['target_id'][0].astype(str)+', '+data_temp['target_name'][0].astype(str)+'\n'+data_temp['production_string'][0].astype(str)+'\n'+hour_str+':'+minute_str+' UTC '+day_str+' '+month_str+' '+year_str+', Orbit '+str(int(data_temp['orbit'][0])),size=30,y=1.01)

    #Time stamp
    plt.text(0.99,0.01,"Created "+str(datetime.datetime.now().day)+' '+calendar.month_abbr[datetime.datetime.now().month]+' '+str(datetime.datetime.now().year)+"\nSource: NASA/JPL-Caltech",ha='right', va='bottom', transform=ax1.transAxes,color='1.0',size=18)

    #Globe inset
    ax2 = inset_axes(ax1,width=2.,height=2.,loc="upper right", axes_class=GeoAxes, axes_kwargs=dict(projection=ccrs.Orthographic(((E+W)/2.),((N+S)/2.))))
    ax2.set_global()
    ax2.scatter(((W+E)/2.),((N+S)/2.),c='r',s=100,marker='*',zorder=3,transform=ccrs.PlateCarree())
    ax2.stock_img()
    ax2.coastlines(color="0.25")

    #Save figure
    print("Saving as "+'OCO3_'+data_temp['production_string'][0].astype(str)+"_"+var_string+'_'+str(data_temp['sounding_id'][0])[:8]+'_'+str(int(data_temp['orbit'][0]))+"_"+data_temp['target_id'][0].astype(str)+'_'+data_temp['target_name'][0].astype(str)+'.png')
    plt.savefig('OCO3_'+data_temp['production_string'][0].astype(str)+"_"+var_string+'_'+str(data_temp['sounding_id'][0])[:8]+'_'+str(int(data_temp['orbit'][0]))+"_"+data_temp['target_id'][0].astype(str)+'_'+data_temp['target_name'][0].astype(str)+'.png')
 
    #Close figure
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot an OCO-3 SAM from a Lite file.")
    parser.add_argument("filename", help="Input OCO-3 Lite file")
    parser.add_argument("site", help="Site to plot'")
    args = parser.parse_args()

    #Load the OCO-3 data from the file
    oco3_data = load_data(args.filename)

    #Plot the SAM of interest
    plot_data(oco3_data, args.site)


if __name__ == "__main__":
    main()


