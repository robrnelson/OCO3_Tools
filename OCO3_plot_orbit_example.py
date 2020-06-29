import numpy as np
import glob
import h5py
import sys
import time
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from pyproj import Proj, transform
import datetime
import calendar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import scipy.cluster.hierarchy as hcluster
from mpl_toolkits.axes_grid1 import make_axes_locatable
from GESDISC_API_Subsetting import subset
import urllib.request
import re
#Can install any missing modules via conda


############################
#Code to plot real OCO-3 TG/AM data!
#Robert R. Nelson, last updated 29 June 2020


def load_orbits(data_inputs):

  data_input = []
  for i in range(len(data_inputs)):
    data_input += sorted(glob.glob(data_inputs[i]))
  data_input_file_names = np.zeros((len(data_input)),dtype='a50')
  for i in range(len(data_input_file_names)):
    data_input_file_names[i] = data_input[i].split('/')[-1]  

  data_input = np.array(data_input)
  data_input_sort_arg = np.argsort(data_input_file_names.astype(str))
  data_input = data_input[data_input_sort_arg]  
  print(len(data_input)," files to load...")

  #Assume a max of ~65,000 soundings per orbit file
  n_soundings_per_orbit = 65000

  #Load L2
  if "L2Std" in data_input[0]:
    data_temp = np.zeros((int(len(data_input) * n_soundings_per_orbit)),dtype=[\
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
        ('sounding_operation_mode','a2'),
        ('sounding_operation_mode_string','a6'),
        ('production_string','a100'),
        ('sounding_pcs_data_source','f8'),
        ('sounding_pcs_data_source_string','a20'),
        ('target_id','a100'),
        ('target_name','a100'),
        ('target_lat','f8'),
        ('target_lon','f8'),
        ('outcome_flag','f8'),
        ('orbit','f8'),
        ('aod_total','f8'),
        ('xco2','f8'),
        ('dp','f8')])

    count, count_new = 0,0

    for i in range(len(data_input)):
      print(data_input[i])
      try: data_file = h5py.File(data_input[i],'r')
      except:
        print("Bad file for some reason...")
        continue

      count_new += len(np.array(data_file['/RetrievalHeader/sounding_id']))

      data_temp['sounding_id'][count:count_new] = np.array(data_file['RetrievalHeader/sounding_id'])
      data_temp['latitude'][count:count_new] = np.array(data_file['RetrievalGeometry/retrieval_latitude'])
      data_temp['longitude'][count:count_new] = np.array(data_file['RetrievalGeometry/retrieval_longitude'])
      data_temp['time'][count:count_new] = np.array(data_file['RetrievalHeader/retrieval_time_tai93'])
      data_temp['outcome_flag'][count:count_new] = np.array(data_file['RetrievalResults/outcome_flag'])
      data_temp['vertex_latitude_1'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_vertex_latitude'])[:,1,0]
      data_temp['vertex_latitude_2'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_vertex_latitude'])[:,1,1]
      data_temp['vertex_latitude_3'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_vertex_latitude'])[:,1,2]
      data_temp['vertex_latitude_4'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_vertex_latitude'])[:,1,3]
      data_temp['vertex_longitude_1'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_vertex_longitude'])[:,1,0]
      data_temp['vertex_longitude_2'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_vertex_longitude'])[:,1,1]
      data_temp['vertex_longitude_3'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_vertex_longitude'])[:,1,2]
      data_temp['vertex_longitude_4'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_vertex_longitude'])[:,1,3]
      data_temp['sounding_operation_mode'][count:count_new] = np.array(data_file['/RetrievalHeader/sounding_operation_mode'])
      try:
        data_temp['sounding_pcs_data_source'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_pcs_data_source'])
      except: print("PMA/PCA data missing...")

      data_temp['xco2'][count:count_new] = np.array(data_file['/RetrievalResults/xco2']) * 1.e6
      data_temp['dp'][count:count_new] = (np.array(data_file['/RetrievalResults/surface_pressure_fph']) - np.array(data_file['RetrievalResults/surface_pressure_apriori_fph'])) / 100. #convert to hPa
      data_temp['aod_total'][count:count_new] = np.array(data_file['AerosolResults/aerosol_total_aod'])

      #Convert sounding_pcs_data_source int flag into a useful string!
      data_temp['sounding_pcs_data_source_string'][count:count_new][data_temp['sounding_pcs_data_source'][count:count_new] == 0] = "SRU+IMU+GPS"
      data_temp['sounding_pcs_data_source_string'][count:count_new][data_temp['sounding_pcs_data_source'][count:count_new] == 1] = "SRU+GPS"
      data_temp['sounding_pcs_data_source_string'][count:count_new][data_temp['sounding_pcs_data_source'][count:count_new] == 2] = "IMU+GPS"
      data_temp['sounding_pcs_data_source_string'][count:count_new][data_temp['sounding_pcs_data_source'][count:count_new] == 3] = "SRU+IMU+BAD"
      data_temp['sounding_pcs_data_source_string'][count:count_new][data_temp['sounding_pcs_data_source'][count:count_new] == 4] = "SRU+BAD"
      data_temp['sounding_pcs_data_source_string'][count:count_new][data_temp['sounding_pcs_data_source'][count:count_new] == 5] = "IMU+BAD"
      data_temp['sounding_pcs_data_source_string'][count:count_new][data_temp['sounding_pcs_data_source'][count:count_new] == 6] = "BAD+GPS"
      data_temp['sounding_pcs_data_source_string'][count:count_new][data_temp['sounding_pcs_data_source'][count:count_new] == 7] = "BAD"

      try: data_temp['orbit'][count:count_new] = data_input[i].rsplit('/')[-1][13:18]
      except: print("Fix this later")

      data_temp['production_string'][count:count_new] = data_file['/Metadata/CollectionLabel']

      data_temp['sounding_operation_mode_string'][count:count_new][data_temp['sounding_operation_mode'][count:count_new] == b'TG'] = "Target"
      data_temp['sounding_operation_mode_string'][count:count_new][data_temp['sounding_operation_mode'][count:count_new] == b'AM'] = "SAM"
      data_temp['sounding_operation_mode_string'][count:count_new][data_temp['sounding_operation_mode'][count:count_new] == b'ND'] = "Nadir"
      data_temp['sounding_operation_mode_string'][count:count_new][data_temp['sounding_operation_mode'][count:count_new] == b'GL'] = "Glint"

      count = count_new
      data_file.close()

    #Crop
    data_temp = data_temp[:count_new]

    #Remove bad outcome flags
    print(data_temp.shape)
    mask_outcome_flag = data_temp['outcome_flag'] <= 2
    data_temp = data_temp[mask_outcome_flag]
    print("Size after filtering on outcome_flag = ",data_temp.shape)

    #Remove sounding_ids that = 0
    mask_temp = data_temp['sounding_id'] != 0
    data_temp = data_temp[mask_temp]
    print("Removed ",np.sum(~mask_temp)," bad sounding_ids")

  else: print("Choose a valid file format...")

  #Remove bad lats
  data_temp = data_temp[(data_temp['latitude'] > -999999.)&(data_temp['latitude'] != 0.0)]

  #Remove bad vertex coordinates
  data_temp = data_temp[(data_temp['vertex_latitude_1'] > -999999.)&(data_temp['vertex_latitude_2'] > -999999.0)&(data_temp['vertex_latitude_3'] > -999999.0)&(data_temp['vertex_latitude_4'] > -999999.0)]

  return data_temp




def get_target_indices(data_temp, out_dir_temp):

  ############################################
  #Find the correct target and its corresponding indices using the SAM/TG list files

  #Grab a .csv of all possible targets from CLASP
  url = "https://oco3car.jpl.nasa.gov/api/report/clasp"
  urllib.request.urlretrieve(url, out_dir_temp+'/oco3_targets_list.csv')
  clasp_data = pd.read_csv(out_dir_temp+'/oco3_targets_list.csv', header=[0])
  os.remove(out_dir_temp+'/oco3_targets_list.csv')

  #Parse the coordinates
  site_center_lat, site_center_lon = np.zeros((len(clasp_data['Site Center WKT']))),np.zeros((len(clasp_data['Site Center WKT'])))
  for i in range(len(clasp_data['Site Center WKT'])):
    site_center_lon[i] = float(str(clasp_data['Site Center WKT'][i]).split(' ')[0][6:])
    site_center_lat[i] = float(str(clasp_data['Site Center WKT'][i]).split(' ')[1][:-1])

  #Load all the SAM/TG list files
  sam_target_list_files_all = sorted(glob.glob("/oco3-p1/scf/oco3_sam_target_list/sam_target_*.txt"))

  #If there are duplicate SAM/TG list files only keep the most recently created
  orbit_temp = np.zeros((len(sam_target_list_files_all)))
  pdt_temp = np.zeros((len(sam_target_list_files_all)))
  for i in range(len(sam_target_list_files_all)):
    orbit_temp[i] = int(re.split('_|\.',os.path.split(sam_target_list_files_all[i])[1])[3])
    pdt_temp[i] = int(re.split('_|\.',os.path.split(sam_target_list_files_all[i])[1])[5])

  sam_target_list_files = []
  orbit_unique,orbit_index,orbit_counts = np.unique(orbit_temp,return_index=True,return_counts=True)
  for i in range(len(orbit_unique)):
    if orbit_counts[i] == 1:
      sam_target_list_files.append(sam_target_list_files_all[orbit_index[i]])
    else:
      pdts = pdt_temp[orbit_temp == orbit_unique[i]]
      sam_target_list_files.append(sam_target_list_files_all[np.where((max(pdts) == pdt_temp)&(orbit_temp == orbit_unique[i]))[0][0]])

  #Load the files!
  sam_target_list_data_temp = []
  for i in range(len(sam_target_list_files)):
    sam_target_list_data_temp.append(np.genfromtxt(sam_target_list_files[i],skip_header=1,unpack=True,usecols=(0,1,3,4,5,6,7),dtype={'names':('orbit','time_stamp','mode_raw','target_id','target_name','site_center_lat','site_center_lon'),'formats':('int','S24','S5','S100','S100','float','float')}))
  sam_target_list_data = np.concatenate(([np.atleast_1d(x) for x in sam_target_list_data_temp]),axis=0)

  #Convert to useful times
  ids_short = np.vectorize(lambda x: x[:14])(data_temp['sounding_id'].astype(int).astype(str))
  oco3_time = pd.to_datetime(ids_short,format='%Y%m%d%H%M%S').astype(np.int64)/10.**9

  sam_target_list_ids_short = np.vectorize(lambda x: x[:19])(sam_target_list_data['time_stamp'].astype(str))
  sam_target_list_time = pd.to_datetime(sam_target_list_ids_short,format='%Y-%m-%dT%H:%M:%S').astype(np.int64)/10.**9 #In seconds

  #Sort all the SAM/TG list commands by time
  time_arg = np.argsort(sam_target_list_time)
  sam_target_list_data = sam_target_list_data[time_arg]
  sam_target_list_time = sam_target_list_time[time_arg]

  #There appear to be duplicate commands, so only look at unique ones (based on the time they were issued)
  unique_time_arg = np.unique(sam_target_list_time,return_index=True)[1]
  sam_target_list_data = sam_target_list_data[unique_time_arg]
  sam_target_list_time = sam_target_list_time[unique_time_arg]

  #Determine indices of individual TG/AMs.
  mask_temp = np.zeros((len(data_temp)))
  mask_temp[data_temp['sounding_operation_mode'] == b'AM'] = 1
  mask_temp[data_temp['sounding_operation_mode'] == b'TG'] = 2
  start_indices_temp = []
  end_indices_temp = []
  target_yes = False

  for i in range(len(data_temp)):
    #When we hit our first "AM" or "TG", consider that a new target
    if (mask_temp[i] == 1) or (mask_temp[i] == 2):
      if target_yes == False:
        start_indices_temp.append(i)
        target_yes = True

        #Look for the most recent SAM/TG list file backwards in time (including if the times are equal!)
        dt = oco3_time[i] - sam_target_list_time
        index = (dt == min(dt[dt >= 0]))

      #For every sounding, use the most recently determined index to assign the SAM/TG list info.
      data_temp['target_id'][i] = sam_target_list_data['target_id'][index][0]
      data_temp['target_name'][i] = sam_target_list_data['target_name'][index][0]

      #Early SAM/TG list files didn't have the target_names listed, so grab them from the master CLASP list
      if data_temp['target_name'][i] == b'none': data_temp['target_name'][i] = np.array(clasp_data['Target Name'])[data_temp['target_id'][i].decode('UTF-8') == np.array(clasp_data['Target ID'])][0].encode('UTF-8')

      #Grab the coordinates of the actual target from CLASP
      data_temp['target_lat'][i] = site_center_lat[data_temp['target_id'][i].decode('UTF-8') == np.array(clasp_data['Target ID'])]
      data_temp['target_lon'][i] = site_center_lon[data_temp['target_id'][i].decode('UTF-8') == np.array(clasp_data['Target ID'])]

      try:
        dt_temp = oco3_time[i] - sam_target_list_time
        index_temp = (dt_temp == min(dt_temp[dt_temp >= 0]))

        #Have we hit a new SAM/TG?
        #1) Is the next OCO3 sounding a different mode?
        #2) Or is the next OCO3 sounding more than 3 minutes past the relevant SAM/TG list command?
        #3) Or is there a different command that's closer previously time to the current sounding. This happens w/ back-to-back commands, BUT need to make sure that the actual start of the new command, designated by an "XS" or a change in mode (e.g. "AM" to "TG"), isn't happening in the next x soundings. Searching the next 50 soundings because the weird SAMs seemed to start ~30 soundings after the time that the command was actually issued. Sometimes this is at the exact same time, but sometimes it's 2+ seconds so we can't simply search on time. Also, add a check that if you find a more recent command in time, that it's actually for a different target. This happens a few times when commands get canceled but reissued later with slightly different start times.

        new_target_mask = (mask_temp[i+1] != mask_temp[i]) |\
                          (oco3_time[i+1] > sam_target_list_time[index]+(60.*3)) |\
                          (((sam_target_list_time[index] - sam_target_list_time[index_temp]) < 0) & (np.all(data_temp['sounding_operation_mode'][i] == data_temp['sounding_operation_mode'][i:i+50])) & (sam_target_list_data['target_id'][index] != sam_target_list_data['target_id'][index_temp]))

        if new_target_mask:
          end_indices_temp.append(i+1)
          target_yes = False

      except:
        print("End of file...")
        end_indices_temp.append(i+1)
        target_yes = False

  return data_temp, start_indices_temp, end_indices_temp

################################################################
def plot_target(data_temp,var_temp,start_indices_temp,end_indices_temp,out_dir_temp):

  #Set some strings
  if var_temp == 'rad_continuum_o2':
    var_string = 'o2_radiance'
    var_cbar_string = 'Radiance [Ph s'+r'$^{-1}$'+' m'+r'$^{-2}$'+' sr'+r'$^{-1}$'+r'$\mu$'+'m'+r'$^{-1}$'+']'
    var_title_string = "O"+r'$_2$'+' A-Band Radiance'
    var_cbar = plt.cm.Purples_r

  elif var_temp == 'xco2':
    var_string = 'xco2'
    var_cbar_string = r'$X_{CO_2}$'+' [ppm]'
    var_title_string = r'$X_{CO_2}$'
    var_cbar = plt.cm.viridis

  elif var_temp == 'dp':
    var_string = 'dp'
    var_cbar_string = 'dp [hPa]'
    var_title_string = 'dp'
    var_cbar = plt.cm.RdBu_r

  elif var_temp == 'aod_total':
    var_string = 'aod'
    var_cbar_string = 'AOD'
    var_title_string = 'AOD'
    var_cbar = plt.cm.magma

  else:
    print("Please select an appropriate variable...")

  ############################################
  #Plot all the individual TG/AMs!
  for i in range(len(start_indices_temp)):
    print("---------------------------------------------------------------")
    print("Plotting TG/AM ",str(i+1)," of ",len(start_indices_temp))
    data_temp2 = data_temp[start_indices_temp[i]:end_indices_temp[i]]
    print(data_temp2['target_id'][0],int(data_temp2['orbit'][0]),data_temp2['sounding_id'][0])

    #Remove weird outliers that are far from the actual TG/AM
    cluster_input = np.array([data_temp2['longitude'],data_temp2['latitude']]).T
    try:
      clusters = hcluster.fclusterdata(cluster_input,1.,criterion='distance')
      print("Cluster #s = ",np.bincount(clusters))

      #Grab the largest cluster, or any/all clusters greater than 30 soundings
      cluster_mask = (clusters == np.argmax(np.bincount(clusters))) | np.isin(clusters, np.where(np.bincount(clusters) > 30)[0])

    except:
      #If there's exactly one sounding, fclusterdata doesn't work
      cluster_mask = np.ones(len(data_temp2),dtype=bool)

    #Determine plotting box
    N,S,E,W = np.round(max(data_temp2['target_lat'][cluster_mask][0],data_temp2['latitude'][cluster_mask].max())+.25,1),\
              np.round(min(data_temp2['target_lat'][cluster_mask][0],data_temp2['latitude'][cluster_mask].min())-.25,1),\
              np.round(max(data_temp2['target_lon'][cluster_mask][0],data_temp2['longitude'][cluster_mask].max())+(.25/np.cos(np.deg2rad(max(data_temp2['target_lat'][cluster_mask][0],data_temp2['latitude'][cluster_mask].max())))),1),\
              np.round(min(data_temp2['target_lon'][cluster_mask][0],data_temp2['longitude'][cluster_mask].min())-(.25/np.cos(np.deg2rad(min(data_temp2['target_lat'][cluster_mask][0],data_temp2['latitude'][cluster_mask].min())))),1)

    #And keep anything within the plotting box
    data_temp2 = data_temp2[(data_temp2['latitude'] < N)&(data_temp2['latitude'] > S)&(data_temp2['longitude']<E)&(data_temp2['longitude']>W)]

    #Get some datetime info from the sounding_id
    month_str = calendar.month_abbr[int(data_temp2['sounding_id'][0].astype(str)[4:6].lstrip("0"))]
    day_str, year_str = data_temp2['sounding_id'][0].astype(str)[6:8].lstrip("0"), data_temp2['sounding_id'][0].astype(str)[:4].lstrip("0")
    hour_str, minute_str = data_temp2['sounding_id'][0].astype(str)[8:10],data_temp2['sounding_id'][0].astype(str)[10:12]

    #Check if the .png exists already. If not, make it!
    if os.path.isfile(out_dir_temp+'/'+data_temp2['production_string'][0].astype(str)+'/'+var_string+'/OCO3_'+data_temp2['production_string'][0].astype(str)+"_"+var_string+'_'+str(data_temp2['sounding_id'][0])[:8]+'_'+str(int(data_temp2['orbit'][0]))+"_"+data_temp2['target_id'][0].astype(str)+'.png'):
      print(out_dir_temp+'/'+data_temp2['production_string'][0].astype(str)+'/'+var_string+'/OCO3_'+data_temp2['production_string'][0].astype(str)+"_"+var_string+'_'+str(data_temp2['sounding_id'][0])[:8]+'_'+str(int(data_temp2['orbit'][0]))+"_"+data_temp2['target_id'][0].astype(str)+'.png already exists!')

    else:
      fig, ax1 = plt.subplots(figsize=[20,20])
      m1 = Basemap(llcrnrlon=W,llcrnrlat=S,urcrnrlon=E,urcrnrlat=N,epsg=3857,resolution='h')
      try: m1.arcgisimage(service='World_Imagery', xpixels = 2000, verbose = True)
      except:
        #Sometimes this fails randomly, so try again
        time.sleep(20)
        m1.arcgisimage(service='World_Imagery', xpixels = 2000, verbose = True)
      m1.drawmeridians(np.linspace(W,E,6)[1:-1],labels=[0,0,0,1],size=24,color='0.75')
      m1.drawparallels(np.linspace(N,S,6)[1:-1],labels=[1,0,0,0],size=24,color='0.75')
      m1.drawcoastlines()

      #Map Scale. Have to hack because the Basemap drawmapscale function is broken, but this should work. Cartopy doesn't even have a map scale function.
      lat0= S+(N-S)*0.04+0.02 #0.04
      lon0= W+(E-W)*0.04
      distance=40./np.cos(lat0*np.pi/180.)
      corner_buffered_m1 = m1(lon0,lat0)
      corner_buffered_lon,corner_buffered_lat = m1(corner_buffered_m1[0] + (distance/2*1000),corner_buffered_m1[1],inverse=True)
      scale=m1.drawmapscale(corner_buffered_lon,corner_buffered_lat,corner_buffered_lon,corner_buffered_lat,distance,units='km',fontcolor='w',fontsize=16,yoffset=2000)
      scale[3].set_text(40)

      #Plot footprints
      x1,y1 = m1(data_temp2['vertex_longitude_1'],data_temp2['vertex_latitude_1'])
      x2,y2 = m1(data_temp2['vertex_longitude_2'],data_temp2['vertex_latitude_2'])
      x3,y3 = m1(data_temp2['vertex_longitude_3'],data_temp2['vertex_latitude_3'])
      x4,y4 = m1(data_temp2['vertex_longitude_4'],data_temp2['vertex_latitude_4'])
      patches = []
      for i in range(len(x1)):
        if (x1[i] == 0.0)|(x2[i] == 0.0)|(x3[i] == 0.0)|(x4[i] == 0.0): print("Bad vertex...")
        else: patches += [Polygon([(x1[i],y1[i]),(x2[i],y2[i]),(x3[i],y3[i]),(x4[i],y4[i])])]
      p = PatchCollection(patches, alpha=1.)

      p.set_array(data_temp2[var_temp])
      if var_temp == 'dp' : p.set_clim(-1*max(np.abs(np.percentile(data_temp2[var_temp],10)),np.abs(np.percentile(data_temp2[var_temp],90))),max(np.abs(np.percentile(data_temp2[var_temp],10)),np.abs(np.percentile(data_temp2[var_temp],90))))
      elif var_temp == 'aod_total': p.set_clim(0,max(0.3,np.percentile(data_temp2[var_temp],90)))
      else: p.set_clim(np.percentile(data_temp2[var_temp],10),np.percentile(data_temp2[var_temp],90))
      p.set_lw(1.0)
      p.set_cmap(var_cbar)
      ax1.add_collection(p)

      #Colorbar
      divider = make_axes_locatable(ax1)
      cax = divider.append_axes("right", size=0.4, pad=0.25)
      if var_temp == 'aod_total': cbar = fig.colorbar(p,extend='max',cax=cax)
      else: cbar = fig.colorbar(p,extend='both',cax=cax)
      cbar.set_label(var_cbar_string,size=28,rotation=270,labelpad=35)
      cbar.ax.tick_params(labelsize=22)
      cbar.ax.yaxis.get_offset_text().set_fontsize(22)

      #Title
      title = ax1.set_title('OCO-3 '+var_title_string+'\n'+data_temp2['sounding_operation_mode_string'][0].astype(str)+' Mode ('+data_temp2['sounding_pcs_data_source_string'][0].astype(str)+'), '+data_temp2['target_id'][0].astype(str)+', "'+data_temp2['target_name'][0].astype(str)+'"\n'+data_temp2['production_string'][0].astype(str)+'\n'+hour_str+':'+minute_str+' UTC '+day_str+' '+month_str+' '+year_str+', Orbit '+str(int(data_temp2['orbit'][0])),size=30,y=1.01)

      #Time stamp
      plt.text(0.99,0.01,"Created "+str(datetime.datetime.now().day)+' '+calendar.month_abbr[datetime.datetime.now().month]+' '+str(datetime.datetime.now().year)+"\nCourtesy NASA/JPL-Caltech (R. R. Nelson)",ha='right', va='bottom', transform=ax1.transAxes,color='1.0',size=18)

      #Preliminary data stamp
      plt.text(0.01,0.99,"PRELIMINARY (vEarly)",ha='left', va='top', transform=ax1.transAxes,color='r',size=18)

      #Globe inset
      ax2 = inset_axes(ax1,width=2.,height=2.)
      m2 = Basemap(projection='ortho',lat_0=((N+S)/2.),lon_0=((E+W)/2.),resolution='l')
      m2.bluemarble()
      x1_globe,y1_globe = m2(((W+E)/2.),((N+S)/2.))
      m2.scatter(x1_globe,y1_globe,c='r',s=100,marker='*')

      #Mark the target
      x1_target,y1_target = m1(data_temp2['target_lon'][0],data_temp2['target_lat'][0])
      ax1.scatter(x1_target,y1_target,c='r',marker="*",s=600)

      #Create save directory if it doesn't exist
      if not os.path.exists(out_dir_temp+'/'+data_temp2['production_string'][0].astype(str)+'/'+var_string):
        os.makedirs(out_dir_temp+'/'+data_temp2['production_string'][0].astype(str)+'/'+var_string)

      #Save figure
      print("Saving as "+out_dir_temp+'/'+data_temp2['production_string'][0].astype(str)+'/'+var_string+'/OCO3_'+data_temp2['production_string'][0].astype(str)+"_"+var_string+'_'+str(data_temp2['sounding_id'][0])[:8]+'_'+str(int(data_temp2['orbit'][0]))+"_"+data_temp2['target_id'][0].astype(str)+'.png')
      #plt.savefig(out_dir_temp+'/'+data_temp2['production_string'][0].astype(str)+'/'+var_string+'/OCO3_'+data_temp2['production_string'][0].astype(str)+"_"+var_string+'_'+str(data_temp2['sounding_id'][0])[:8]+'_'+str(int(data_temp2['orbit'][0]))+"_"+data_temp2['target_id'][0].astype(str)+'.png')

      #Close figure
      #plt.close()


########################################################
if __name__ == "__main__":

  out_dir = '~/out_dir/'

  #Load the data. The paths get globbed and then combined, so you can be clever on exactly what data you want to read in
  #All 28 Feb. 2020 OCO-3 orbits from SCF_B10110_r01!
  data_l2 = load_orbits(['/oco3-p2/scf/SCF_B10110_r01/2020/02/28/L2Std/oco3_L2StdSC_*.h5'])

  #Get indices of SAMs/TGs
  data_l2, start_indices, end_indices = get_target_indices(data_l2,out_dir)

  #Plot XCO2 maps
  plot_target(data_l2,"xco2",start_indices,end_indices,out_dir)



