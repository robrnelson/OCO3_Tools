import numpy as np
import glob
import h5py
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from pyproj import Proj, transform
import datetime
import calendar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.cluster.hierarchy as hcluster 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from haversine import haversine #Can install via conda

#Code to plot real OCO-3 TG/AM data!
#Robert R. Nelson, last updated 19 September 2019

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

  #Load L2
  if ("l2_plus_more" in data_input[0]) | ("L2Std" in data_input[0]):
    data_temp = np.zeros((3000000),dtype=[\
	('sounding_id','int'),
	('latitude','f8'),
	('longitude','f8'),
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
	('solar_zenith_angle','f8'),
	('production_string','a20'),
	('sounding_pcs_mode','a2'),
	('sounding_pcs_data_source','f8'),
	('sounding_pcs_data_source_string','a5'),
	('outcome_flag','f8'),
	('orbit','f8'),
	('albedo_o2','f8'),
	('albedo_wco2','f8'),
	('albedo_sco2','f8'),
	('albedo_slope_o2','f8'),
	('albedo_slope_wco2','f8'),
	('albedo_slope_sco2','f8'),
	('land_fraction','f8'),
	('polarization_angle','f8'),
	('chi2_o2','f8'),
	('chi2_wco2','f8'),
	('chi2_sco2','f8'),
	('sounding_index','f8'),
	('surface_type','f8'),
	('co2_ratio','f8'),
	('snr_o2','f8'),
	('snr_wco2','f8'),
	('snr_sco2','f8'),
	('xco2','f8'),
	('iterations','f8'),
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
      data_temp['outcome_flag'][count:count_new] = np.array(data_file['RetrievalResults/outcome_flag'])
      data_temp['vertex_latitude_1'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_vertex_latitude'])[:,1,0]
      data_temp['vertex_latitude_2'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_vertex_latitude'])[:,1,1]
      data_temp['vertex_latitude_3'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_vertex_latitude'])[:,1,2]
      data_temp['vertex_latitude_4'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_vertex_latitude'])[:,1,3]
      data_temp['vertex_longitude_1'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_vertex_longitude'])[:,1,0]
      data_temp['vertex_longitude_2'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_vertex_longitude'])[:,1,1]
      data_temp['vertex_longitude_3'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_vertex_longitude'])[:,1,2]
      data_temp['vertex_longitude_4'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_vertex_longitude'])[:,1,3]
      data_temp['xco2'][count:count_new] = np.array(data_file['/RetrievalResults/xco2']) * 1.e6
      data_temp['sounding_operation_mode'][count:count_new] = np.array(data_file['/RetrievalHeader/sounding_operation_mode'])
      data_temp['sounding_index'][count:count_new] = np.array(data_file['/RetrievalHeader/sounding_index'])
      data_temp['sounding_pcs_data_source'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_pcs_data_source'])
      data_temp['sounding_pcs_mode'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_pcs_mode'])
      data_temp['polarization_angle'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_polarization_angle'])
      data_temp['solar_zenith_angle'][count:count_new] = np.array(data_file['/RetrievalGeometry/retrieval_solar_zenith'])
      data_temp['surface_type'][count:count_new][np.array(data_file['/RetrievalResults/surface_type']) == b'Coxmunk,Lambertian'] = 0
      data_temp['surface_type'][count:count_new][np.array(data_file['/RetrievalResults/surface_type']) == b'BRDF Soil         '] = 1
      data_temp['chi2_o2'][count:count_new] = np.array(data_file['/SpectralParameters/reduced_chi_squared_o2_fph'])
      data_temp['chi2_wco2'][count:count_new] = np.array(data_file['/SpectralParameters/reduced_chi_squared_weak_co2_fph'])
      data_temp['chi2_sco2'][count:count_new] = np.array(data_file['/SpectralParameters/reduced_chi_squared_strong_co2_fph'])
      data_temp['snr_o2'][count:count_new] = np.array(data_file['/SpectralParameters/signal_o2_fph']) /  np.array(data_file['/SpectralParameters/noise_o2_fph'])
      data_temp['snr_wco2'][count:count_new] = np.array(data_file['/SpectralParameters/signal_weak_co2_fph']) /  np.array(data_file['/SpectralParameters/noise_weak_co2_fph'])
      data_temp['snr_sco2'][count:count_new] = np.array(data_file['/SpectralParameters/signal_strong_co2_fph']) /  np.array(data_file['/SpectralParameters/noise_strong_co2_fph'])
      data_temp['iterations'][count:count_new] = np.array(data_file['/RetrievalResults/iterations'])
      data_temp['dp'][count:count_new] = np.array(data_file['/RetrievalResults/surface_pressure_fph']) - np.array(data_file['RetrievalResults/surface_pressure_apriori_fph'])
      data_temp['co2_ratio'][count:count_new] = np.array(data_file['/PreprocessingResults/co2_ratio_idp'])

      #Albedos
      land_mask_temp = np.array(data_file['/RetrievalResults/surface_type']) == b'BRDF Soil         ' #String nonsense
      try:
        data_temp['albedo_o2'][count:count_new][land_mask_temp] = np.array(data_file['/BRDFResults/brdf_reflectance_o2'])[land_mask_temp]
        data_temp['albedo_wco2'][count:count_new][land_mask_temp] = np.array(data_file['/BRDFResults/brdf_reflectance_weak_co2'])[land_mask_temp]
        data_temp['albedo_sco2'][count:count_new][land_mask_temp] = np.array(data_file['/BRDFResults/brdf_reflectance_strong_co2'])[land_mask_temp]
      except: print("No land data in this file!")
      try:
        data_temp['albedo_o2'][count:count_new][~land_mask_temp] = np.array(data_file['/AlbedoResults/albedo_o2_fph'])[~land_mask_temp]
        data_temp['albedo_wco2'][count:count_new][~land_mask_temp] = np.array(data_file['/AlbedoResults/albedo_weak_co2_fph'])[~land_mask_temp]
        data_temp['albedo_sco2'][count:count_new][~land_mask_temp] = np.array(data_file['/AlbedoResults/albedo_strong_co2_fph'])[~land_mask_temp]
      except: print("No ocean data in this file!")

      #Albedo slopes
      try:
        data_temp['albedo_slope_o2'][count:count_new][land_mask_temp] = np.array(data_file['/BRDFResults/brdf_reflectance_slope_o2'])[land_mask_temp]
        data_temp['albedo_slope_wco2'][count:count_new][land_mask_temp] = np.array(data_file['/BRDFResults/brdf_reflectance_slope_weak_co2'])[land_mask_temp]
        data_temp['albedo_slope_sco2'][count:count_new][land_mask_temp] = np.array(data_file['/BRDFResults/brdf_reflectance_slope_strong_co2'])[land_mask_temp]
      except: print("No land data in this file!")
      try:
        data_temp['albedo_slope_o2'][count:count_new][~land_mask_temp] = np.array(data_file['/AlbedoResults/albedo_slope_o2'])[~land_mask_temp]
        data_temp['albedo_slope_wco2'][count:count_new][~land_mask_temp] = np.array(data_file['/AlbedoResults/albedo_slope_weak_co2'])[~land_mask_temp]
        data_temp['albedo_slope_sco2'][count:count_new][~land_mask_temp] = np.array(data_file['/AlbedoResults/albedo_slope_strong_co2'])[~land_mask_temp]
      except: print("No ocean data in this file!")

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

    #Remove bad lats
    mask_temp = (data_temp['latitude'] > -999999)&(data_temp['latitude'] != 0.0)
    data_temp = data_temp[mask_temp]
    print("Removed ",np.sum(~mask_temp)," bad lats")

  else: print("Choose a valid file format...")

  return data_temp





############################
#Load the data. The paths get globbed and then combined, so you can be clever on exactly what data you want to read in

#All September 13th orbits!
data_l2 = load_orbits(['/oco3-p2/scf/SCF_B9210_r01/2019/09/13/L2Std/oco3_L2StdSC_*.h5'])



############################
def plot_map(data_temp):

  #Determine indices of individual TG/AMs.
  mask_temp = np.zeros((len(data_temp)))
  mask_temp[data_temp['sounding_operation_mode'] == b'AM'] = 1
  mask_temp[data_temp['sounding_operation_mode'] == b'TG'] = 2
  start_indices = []
  end_indices = []
  target_yes = False
  for i in range(len(data_temp)):
    #When we hit our first "AM" or "TG", consider that a new target
    if (mask_temp[i] == 1) or (mask_temp[i] == 2):
      if target_yes == False:
        start_indices.append(i)
        target_yes = True
      try:
        #When we stop recording consecutive AMs or TGs, end the target
        if mask_temp[i+1] != mask_temp[i]:
          end_indices.append(i+1)
          target_yes = False
      except: 
        print("End of file...")
        end_indices.append(i+1)
        target_yes = False

  #Grab all the TG/AM names from a csv
  target_id, target_name, site_center, mode_raw = np.genfromtxt('/data/oco3_targets_20190918_cropped.csv',skip_header=1,unpack=True,delimiter=',',usecols=(0,1,3,4),dtype=None)
  site_center_lat, site_center_lon = np.zeros((len(site_center))),np.zeros((len(site_center)))
  for i in range(len(site_center)):
    site_center_lon[i] = float(str(site_center[i]).split(' ')[0][8:])
    site_center_lat[i] = float(str(site_center[i]).split(' ')[1][:-2])


  ############################
  epsg = 3857 #Web mercator, fine for small areas
  cbar_label_size = 16
  scat_size = 18
  dref=40. #km, for plotting a scale

  #Plot all the individual TG/AMs!
  for i in range(len(start_indices)):
    print("Plotting TG/AM ",str(i+1)," of ",len(start_indices))
    data_temp2 = data_temp[start_indices[i]:end_indices[i]]

    #Remove weird outliers that are far from the actual TG/AM by selecing the main "cluster" of soundings
    cluster_input = np.array([data_temp['longitude'][start_indices[i]:end_indices[i]],data_temp['latitude'][start_indices[i]:end_indices[i]]]).T
    clusters = hcluster.fclusterdata(cluster_input,0.5,criterion='distance')
    if np.sum(np.bincount(clusters) > 30) > 1:
      print("There are multiple large clusters of data, which means something has gone wrong.")
      sys.exit()
    main_cluster_index = np.argmax(np.bincount(clusters))
    data_temp2 = data_temp2[clusters == main_cluster_index]

    #Automatically set the map bounds
    N,S,E,W = np.round(data_temp2['latitude'].max()+.25,1),np.round(data_temp2['latitude'].min()-.25,1),np.round(data_temp2['longitude'].max()+(.25/np.cos(np.deg2rad(data_temp2['latitude'].max()))),1),np.round(data_temp2['longitude'].min()-(.25/np.cos(np.deg2rad(data_temp2['latitude'].min()))),1)

    #Create figure
    fig, ax1 = plt.subplots(figsize=[20,20])
    m1 = Basemap(llcrnrlon=W,llcrnrlat=S,urcrnrlon=E,urcrnrlat=N,epsg=epsg,resolution='h')
    m1.arcgisimage(service='World_Imagery', xpixels = 2000, verbose = True)
    m1.drawmeridians(np.linspace(W,E,6)[1:-1],labels=[0,0,0,1],size=24,color='0.75')
    m1.drawparallels(np.linspace(N,S,6)[1:-1],labels=[1,0,0,0],size=24,color='0.75')
    m1.drawcoastlines()

    #Map Scale. Have to hack because the Basemap drawmapscale function is broken, but this should work. Cartopy doesn't even have a map scale function.
    lat0,lon0= S+(N-S)*0.04, W+(E-W)*0.04
    distance=dref/np.cos(lat0*np.pi/180.)
    corner_buffered_m1 = m1(lon0,lat0)
    corner_buffered_lon,corner_buffered_lat = m1(corner_buffered_m1[0] + (distance/2*1000),corner_buffered_m1[1],inverse=True)
    scale=m1.drawmapscale(corner_buffered_lon,corner_buffered_lat,corner_buffered_lon,corner_buffered_lat,distance,units='km',fontcolor='w',fontsize=16,yoffset=2000)
    scale[3].set_text(dref)

    #Plot individual footprints
    x1,y1 = m1(data_temp2['vertex_longitude_1'],data_temp2['vertex_latitude_1'])
    x2,y2 = m1(data_temp2['vertex_longitude_2'],data_temp2['vertex_latitude_2'])
    x3,y3 = m1(data_temp2['vertex_longitude_3'],data_temp2['vertex_latitude_3'])
    x4,y4 = m1(data_temp2['vertex_longitude_4'],data_temp2['vertex_latitude_4'])
    patches = []
    for i in range(len(x1)):
      if (x1[i] == 0.0)|(x2[i] == 0.0)|(x3[i] == 0.0)|(x4[i] == 0.0): print("Bad vertex...")
      else: patches += [Polygon([(x1[i],y1[i]),(x2[i],y2[i]),(x3[i],y3[i]),(x4[i],y4[i])])]

    p = PatchCollection(patches, alpha=1.)
    p.set_array(data_temp2['xco2'])

    #Set the colorbar limits automatically
    p.set_clim(np.percentile(data_temp2['xco2'],10),np.percentile(data_temp2['xco2'],90))

    #Set the colormap
    p.set_cmap(plt.cm.viridis)

    ax1.add_collection(p)

    #Colorbar
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size=0.4, pad=0.25)
    cbar = fig.colorbar(p,extend='both',cax=cax)
    cbar.set_label(r'$X_{CO_2}$'+' [ppm]',size=28,rotation=270,labelpad=35)
    cbar.ax.tick_params(labelsize=22)

    #Do some annoying stuff to make a time stamp
    month_str = calendar.month_abbr[int(data_temp2['sounding_id'][0].astype(str)[4:6].lstrip("0"))]
    day_str, year_str = data_temp2['sounding_id'][0].astype(str)[6:8].lstrip("0"), data_temp2['sounding_id'][0].astype(str)[:4].lstrip("0")
    hour_str, minute_str = data_temp2['sounding_id'][0].astype(str)[8:10],data_temp2['sounding_id'][0].astype(str)[10:12]

    #Find the closest listed target in space and use that as a label. There really needs to be a better way to do this without having to load auxiliary files
    dist_temp = np.zeros((len(target_id)))
    mean_lat_temp,mean_lon_temp = np.median(data_temp2['latitude']), np.median(data_temp2['longitude']) 
    for j in range(len(dist_temp)):
      dist_temp[j] = haversine((mean_lat_temp,mean_lon_temp),(site_center_lat[j],site_center_lon[j]))
    target_id_temp, target_name_temp = target_id[np.argmin(dist_temp)].astype(str), target_name[np.argmin(dist_temp)].astype(str)
    site_center_lat_temp, site_center_lon_temp = site_center_lat[np.argmin(dist_temp)],site_center_lon[np.argmin(dist_temp)] 

    #Lengthly titles
    title = ax1.set_title('OCO-3 '+r'$X_{CO_2}$'+'\n'+data_temp2['sounding_operation_mode_string'][0].astype(str)+' Mode ('+data_temp2['sounding_pcs_data_source_string'][0].astype(str)+'), '+target_id_temp+', "'+target_name_temp+'"\n'+data_temp2['production_string'][0].astype(str)+', '+hour_str+':'+minute_str+' UTC '+day_str+' '+month_str+' '+year_str+', Orbit '+str(int(data_temp2['orbit'][0])),size=30,y=1.01)

    #Time stamp
    plt.text(0.99,0.03,"Created "+str(datetime.datetime.now().day)+' '+calendar.month_abbr[datetime.datetime.now().month]+' '+str(datetime.datetime.now().year),ha='right', va='top', transform=ax1.transAxes,color='1.0',size=18)

    #Globe inset
    ax2 = inset_axes(ax1,width=2.,height=2.)
    m2 = Basemap(projection='ortho',lat_0=((N+S)/2.),lon_0=((E+W)/2.),resolution='l')
    m2.bluemarble()
    x1_globe,y1_globe = m2(((W+E)/2.),((N+S)/2.))
    m2.scatter(x1_globe,y1_globe,c='#dadaeb',s=100,marker='*')

    #Put a red cross where the target is
    x1_target,y1_target = m1(site_center_lon_temp,site_center_lat_temp)
    ax1.scatter(x1_target,y1_target,c='r',marker="+",s=300)

    #Save!
    #print("Saving as ",data_temp2['production_string'][0].astype(str)+'/.../OCO3_..._'+str(data_temp2['sounding_id'][0])[:8]+'_'+target_id_temp+'.png')
    #plt.savefig('~/'+data_temp2['production_string'][0].astype(str)+'/xco2/OCO3_xco2_'+str(data_temp2['sounding_id'][0])[:8]+'_'+target_id_temp+'.png')
    #plt.close()


#########################
plot_map(data_l2)




