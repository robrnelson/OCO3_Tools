import numpy  as np
import h5py
import pylab as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
from matplotlib import ticker, cm
import pandas as pd
from PIL import Image
import os
import urllib
import datetime
import calendar
from pyproj import Proj, transform 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition


def get_xco2_from_emission(INemission,INmonth,INboundarylayerheight):

    monthday = [31,28,31,30,31,30,31,31,30,31,30,31]

    Av = 6.023e23 # Avagrado's number molecules/mole

    # number density of air
    rho_air = 2.69e25                       # molec/m^3 NOTE FIXED NEEDS ADJUSTING RE SURFACE PRESSURE
    rho_air = np.multiply(rho_air,1e9)      # molec/km^3    
    
    # NOTE: ODIAC described as tonnes C/1km box/month
    
    INemission = np.multiply(INemission,44/12.)              # tonnes CO2/1 km box/month
    dayhourconversion = monthday[INmonth-1]*24               # days/month * hours/day = hours/month
    INemission = np.divide(INemission,dayhourconversion)     # tonnes CO2/1 km box/hour
    INemission = np.multiply(INemission,1e6)                 # grams CO2/1 km box/hour
    INemission = np.divide(INemission,44)                    # mole CO2/1 km box/hour
    INemission = np.multiply(INemission,Av)                  # molec CO2/1 km box/hour
    INemission = np.divide(INemission,INboundarylayerheight) # molec CO2/km^3

    return np.divide(INemission,rho_air)
    

def readodiac(extent,monthchar):
    
    #filename = 'odiac2018/1km_tif/odiac2018_1km_excl_intl_17'+monthchar+'.tif'
    filename = '/scratch-science2/algorithm/rnelson/ODIAC/odiac2018_1km_excl_intl_17'+monthchar+'.tif'

    import tifffile as tf

    image_stack = tf.imread(filename)
    #print(image_stack.shape)
    #print(image_stack.dtype)

    nrows = len(image_stack)
    ncols = len(image_stack[1])

    Unit  = 'Tonne Carbon/cell/month'

    dx = dy = 0.008333333333333

    ODIAClons = np.arange(ncols)*dx + dx/2. - 180.
    ODIAClats = 90 - np.arange(nrows)*dy + dy/2. 

    ind = np.where(image_stack == 0.)
    image_stack[ind] = np.nan

    #Focus on target region

    indlon = np.where((ODIAClons >= extent[0]) & (ODIAClons <= extent[1]))
    indlat = np.where((ODIAClats >= extent[2]) & (ODIAClats <= extent[3]))

    image_stack = np.squeeze(image_stack[:,indlon])
    image_stack = np.squeeze(image_stack[indlat,:])

    return image_stack, ODIAClons[indlon], ODIAClats[indlat]



#Load all possible targets
target_id, target_name, site_center, mode_raw = np.genfromtxt('/home/rnelson/data/OCO3/oco3_targets_20190918_cropped.csv',skip_header=1,unpack=True,delimiter=',',usecols=(0,1,3,4),dtype=None)
site_center_lat, site_center_lon = np.zeros((len(site_center))),np.zeros((len(site_center)))
for i in range(len(site_center)):
  site_center_lon[i] = float(str(site_center[i]).split(' ')[0][8:])
  site_center_lat[i] = float(str(site_center[i]).split(' ')[1][:-2])


#Turn off interactive plotting if you're gonna make a few hundred plots
#plt.ioff()

#Loop through all the potential OCO-3 target locations!
for i in range(len(site_center_lon)):

  print("Plotting ",target_id[i].astype(str),"/",target_name[i].astype(str))

  extent = [site_center_lon[i]-1,
            site_center_lon[i]+1,
            site_center_lat[i]-1,
            site_center_lat[i]+1]

  #-------------------------
  # Read ODIAC data
  #-------------------------

  co2flux, odiaclons, odiaclats = readodiac(extent,'01')

  #-------------------------
  # Plot ODIAC data
  #-------------------------

  X, Y = np.meshgrid(odiaclons,odiaclats)

  #odiacdatapoints = np.count_nonzero(~np.isnan(co2flux))
  #if odiacdatapoints == 0:

  central_lat = 0
  central_lon = 0
  proj = ccrs.PlateCarree(central_longitude=central_lon)

  fig1, (ax1,ax2) = plt.subplots(1,2, sharex=True, sharey=True, figsize=(20,8), subplot_kw={'projection': proj})
  fig1.subplots_adjust(wspace=.15)

  ax1.set_extent(extent)

  N,S,E,W = extent[3],extent[2],extent[1],extent[0]
  W_3857, S_3857 = transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), W,S)
  E_3857, N_3857 = transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), E,N)

  url = 'http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox='+str(W)+','+str(S)+','+str(E)+','+str(N)+'&bboxSR=4326&imageSR=4326&size=500,500,&dpi=96&format=png32&transparent=true&f=image'
  ESRI = Image.open(urllib.request.urlopen(url))
  im1 = ax1.imshow(ESRI,extent=extent,origin='upper',transform=ccrs.PlateCarree())

  try: 
    surf = ax1.contourf(X,Y,co2flux,locator=ticker.LogLocator(),transform=ccrs.PlateCarree(),cmap=plt.cm.Reds)
    cb = plt.colorbar(surf,ax=ax1, orientation='vertical', label = 'ODIAC CO$_2$ emissions [Tonne C/cell/month]')
  except: print("Contourf failed, probably because the co2flux values were only in 1 log bin and it doesn't like that...")

  gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1.5, color='gray')


  #Coastlines
  ax1.coastlines(resolution='10m', color='black', linewidth=1)

  #Globe inset
  ax3 = plt.axes([0,0,1,1], projection=ccrs.Orthographic(central_longitude=site_center_lon[i],central_latitude=site_center_lat[i]))
  ip1 = InsetPosition(ax1, [0.78, 0.78, 0.2, 0.2])
  ax3.set_axes_locator(ip1)
  ax3.set_global()
  ax3.add_feature(cfeature.LAND)
  ax3.add_feature(cfeature.OCEAN)
  ax3.add_feature(cfeature.COASTLINE)
  ax3.scatter(site_center_lon[i],site_center_lat[i],c='r',marker="+",s=300)

  gl.xlabels_top = False
  gl.ylabels_right = False

  ####################################
  # Plot ODIAC XCO2 after making some assumptions

  usemonth               = 8 # August
  useboundarylayerheight = 2 # km

  #print(np.nanmin(co2flux),np.nanmax(co2flux),np.nanmedian(co2flux))
  
  odiac_xco2 = get_xco2_from_emission(co2flux,usemonth,useboundarylayerheight)
  odiac_xco2 = np.divide(odiac_xco2,1e-6)

  v_min = 0
  v_max = 3 #5
  npts  = 100
  uselevels = np.arange(npts)*(v_max-v_min)/npts

  ax2.set_extent(extent)
  im2 = ax2.imshow(ESRI,extent=extent,origin='upper',transform=ccrs.PlateCarree())
  
  surfxco2 = ax2.pcolormesh(X,Y,odiac_xco2, transform=ccrs.PlateCarree(), vmin=v_min,vmax=v_max, cmap=plt.cm.viridis)
  #surfxco2 = ax2.contour(X,Y,odiac_xco2, transform=ccrs.PlateCarree(), vmin=v_min,vmax=v_max, cmap=plt.cm.viridis)
  #surfxco2 = ax2.contourf(X,Y,odiac_xco2,levels=uselevels, transform=ccrs.PlateCarree(), vmin=v_min,vmax=v_max, cmap=plt.cm.viridis)

  gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1.5, color='gray')
  cb2 = plt.colorbar(surfxco2, ax=ax2, extend='max', orientation='vertical', label='ODIAC '+r'$X_{CO_2}$'+' Enhancement [ppm]')
  #cb2 = plt.colorbar(surfxco2, ax=ax2, extend='max', orientation='vertical', label='ODIAC boundary layer '+r'$X_{CO_2}$'+' [ppm]')

  gl2.xlabels_top = False
  gl2.ylabels_right = False

  #Coastlines 
  ax2.coastlines(resolution='10m', color='black', linewidth=1)

  #Target markers
  ax1.scatter(site_center_lon[i],site_center_lat[i],c='w',marker="+",s=300)
  ax2.scatter(site_center_lon[i],site_center_lat[i],c='w',marker="+",s=300)


  #Globe inset (oddly, matplotlib complains if I try and create this new axis with the same location as ax3)
  ax4 = plt.axes([1,1,1,1], projection=ccrs.Orthographic(central_longitude=site_center_lon[i],central_latitude=site_center_lat[i]))
  ip2 = InsetPosition(ax2, [0.78, 0.78, 0.2, 0.2])
  ax4.set_axes_locator(ip2)
  ax4.set_global()
  ax4.add_feature(cfeature.LAND)
  ax4.add_feature(cfeature.OCEAN)
  ax4.add_feature(cfeature.COASTLINE)
  ax4.scatter(site_center_lon[i],site_center_lat[i],c='r',marker="+",s=300)
  
  #Timestamp
  ax1.text(0.99,0.04,"Created "+str(datetime.datetime.now().day)+' '+calendar.month_abbr[datetime.datetime.now().month]+' '+str(datetime.datetime.now().year),ha='right', va='top', transform=ax1.transAxes,color='1.0',size=10)
  ax2.text(0.99,0.04,"Created "+str(datetime.datetime.now().day)+' '+calendar.month_abbr[datetime.datetime.now().month]+' '+str(datetime.datetime.now().year),ha='right', va='top', transform=ax2.transAxes,color='1.0',size=10)

  #Title
  title = plt.suptitle('ODIAC Emissions for '+target_id[i].astype(str)+' ('+target_name[i].astype(str)+")",size=20,y=0.95)

  plt.show()
  #plt.savefig('/home/rnelson/figures/ACOS/OCO-3/OCO3_ODIAC_aspect_test/OCO3_ODIAC_'+target_id[i].astype(str)+'.png')
  #plt.savefig('/home/rnelson/figures/ACOS/OCO-3/OCO3_ODIAC/OCO3_ODIAC_'+target_id[i].astype(str)+'.png')
  #plt.close()



