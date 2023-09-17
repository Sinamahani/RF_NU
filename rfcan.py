from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import obspy
from obspy.taup import TauPyModel
import os
import glob
from rfpy import RFData
from os.path import exists
import numpy as np
import shutil
model = TauPyModel(model="iasp91")
client = Client("IRIS")
import matplotlib.pyplot as plt
from matplotlib.dates import num2date
plt.style.use('ggplot')
import pickle


############################################################################################################################################################################
# Class for making database
############################################################################################################################################################################
class make_db:
    """
    This class is written for making database of stations and events and waveforms.
    for using this, you need to define a dictionary with following keys:
    
    db_name: main directory name
    network_list: list of networks
    station_list: list of stations
    starttime: start time
    endtime: end time
    location_box: [min latitude, max latitude, min longitude, max longitude]
    event_radius: [center latitude, center longitude, min radius, max radius]
    minmag: minimum magnitude
    maxmag: maximum magnitude
    """
#############################################################################################################################################################################
    def __init__(self, input_values):
        self.db_name = input_values["db_name"]                                              # main directory name
        self.net_list = input_values["network_list"]                                        # network list
        self.sta = input_values["station"]                                        # station list
        self.stime = input_values["starttime"]                                              # start time
        self.etime = input_values["endtime"]                                                # end time
        self.minla = input_values["location_box"][0]                                        # min latitude
        self.maxla = input_values["location_box"][1]                                        # max latitude
        self.minlo = input_values["location_box"][2]                                        # min longitude
        self.maxlo = input_values["location_box"][3]                                        # max longitude
        self.cenla = input_values["event_radius"][0]                                        # center latitude
        self.cenlo = input_values["event_radius"][1]                                        # center longitude
        self.minrad = input_values["event_radius"][2]                                       # min radius
        self.maxrad = input_values["event_radius"][3]                                       # max radius
        self.minmag = input_values["minmag"]                                                # minimum magnitude
        self.maxmag = input_values["maxmag"]                                                # maximum magnitude
        self.siglen = input_values["siglen"]                                                # signal length

        # Creating main directory
        self.create_main_directory_path()
        self.get_stations()
        self.get_events()
        


########################################################################################################################################################################
    def create_main_directory_path(self):
        if not os.path.exists(self.db_name):
            os.makedirs(self.db_name)


########################################################################################################################################################################
    def get_stations(self):
        """
        This function is written for getting stations from IRIS and writing them in a xml file.
        """
        inv_path = glob.glob(f"{self.db_name}/{self.sta}/inventory_*.xml")
        if len(inv_path) > 0:
            if os.path.isfile(inv_path[0]):
                inventory = obspy.read_inventory(f"{self.db_name}/{self.sta}/inventory_*.xml", format="STATIONXML")
                print("Inventory is already downloaded.")
        else:
            inventory = client.get_stations(network = ",".join(self.net_list), station = ",".join(self.sta), minlatitude=self.minla, maxlatitude=self.maxla, minlongitude=self.minlo, maxlongitude=self.maxlo
                                                    ,starttime=self.stime,endtime=self.etime)
            inventory.write(f"{self.db_name}/{self.sta}/inventory_{','.join(self.net_list)}_{self.stime.year}-{self.etime.year}.xml", format="STATIONXML")
            print("Inventory is downloaded now.")
        self.inventory = inventory
        print("# of Stations:", (self.station_counter()))
        
            

########################################################################################################################################################################
    def station_counter(self):
        counter = 0
        for net in self.inventory:
            for sta in net:
                counter += 1
        return counter


########################################################################################################################################################################
    def get_events(self):
        """
        This function is written for getting events from IRIS and writing them in a xml file.
        """
        ev_path = glob.glob(f"{self.db_name}/{self.sta}/catalog_*.xml")
        if len(ev_path) > 0:
            if os.path.isfile(ev_path[0]):
                catal = obspy.read_events(f"{self.db_name}/{self.sta}/catalog_*.xml")
                print("Catalog is already downloaded.")
        else:
            catal = client.get_events(starttime=self.stime,endtime=self.etime, latitude = self.cenla, longitude = self.cenlo,
                                        minradius = self.minrad, maxradius = self.maxrad, minmagnitude = self.minmag, maxmagnitude = self.maxmag)                    
            catal.write(f"{self.db_name}/{self.sta}/catalog_{','.join(self.net_list)}_{self.stime.year}-{self.etime.year}.xml", format="QUAKEML")
            print("Catalog is downloaded now.")
        self.catal = catal
        print("# of Events:", len(catal))

########################################################################################################################################################################
    def create_sub_directories(self, name):
        if not os.path.exists(f"{self.db_name}/{self.sta}/{name}"):
            os.makedirs(f"{self.db_name}/{self.sta}/{name}")


########################################################################################################################################################################
    def get_waveforms_bulk(self):
        """
        This function is written for getting waveforms' info from IRIS and writing them in a bulk file.
        """
        
        bulk = []    # an empty bulk list for downloading waveforms
        for network in self.inventory:
            for station in network:
                for event in self.catal:
                    event_time = event.origins[0].time
                    if station.is_active(event_time):       #check if the channel is active at the time of the event
                        bulk.append([network.code, station.code, "*", "BH?", event_time, event_time + self.siglen])
                        bulk.append([network.code, station.code, "*", "HH?", event_time, event_time + self.siglen])
                        self.create_sub_directories(f"{self.sta}/DATA/{event_time.year}_{event_time.month}_{event_time.day}.{event_time.hour}_{event_time.minute}_{event_time.second}_")
                        event.write(f"{self.db_name}/{self.sta}/DATA/{event_time.year}_{event_time.month}_{event_time.day}.{event_time.hour}_{event_time.minute}_{event_time.second}_/event.xml", format="QUAKEML")
                        
        self.bulk = np.array(bulk)
        self.create_sub_directories("bulk_downloads")
        
        #detrmining the batch size for bulk downloading
        if len(self.bulk) < 200:
            self.bulk_batch_size = 1
        elif len(self.bulk) < 800:
            self.bulk_batch_size = 3
        else:
            self.bulk_batch_size = 25

        bulk_download_status = np.hstack((bulk, np.zeros((len(self.bulk),1))))
        num_of_batches = np.int_(np.linspace(0, len(bulk_download_status), self.bulk_batch_size+1))
        for i in range(len(num_of_batches)-1):
            current_batch = bulk_download_status[num_of_batches[i]:num_of_batches[i+1]]
            try:  
                mini_bulk_download = client.get_waveforms_bulk(current_batch[:,:-1].tolist())
                mini_bulk_download.write(f"{self.db_name}/bulk_downloads/bulk_{i}.sac", format="sac")
                bulk_download_status[num_of_batches[i]:num_of_batches[i+1], -1] = "OK"
            except:
                print(f"Error in downloading bulk files. --> bulk_{i+1}")
                bulk_download_status[num_of_batches[i]:num_of_batches[i+1], -1] = "problem"
            print(f"Progress in bulk downloading: {round((i+1)/(len(num_of_batches)-1)*100,0)} %", end='\r')
        self.bulk = bulk_download_status.tolist()
        np.savetxt(f"{self.db_name}/01_bulk_file.txt", self.bulk, fmt="%s") 

        self.distribute_bulk_waveforms()


########################################################################################################################################################################
    def distribute_bulk_waveforms(self):
        
        # try:
        waveforms = obspy.read(f"{self.db_name}/bulk_downloads/*.sac")
        for tr in waveforms:
            network = tr.stats.network
            station = tr.stats.station
            channel = tr.stats.channel
            location = tr.stats.location
            year = tr.stats.starttime.year
            month = tr.stats.starttime.month
            day = tr.stats.starttime.day
            hour = tr.stats.starttime.hour
            minute = tr.stats.starttime.minute
            second = tr.stats.starttime.second
            try:
                # print(f"{self.db_name}/DATA/{year}_{month}_{day}.{hour}_{minute}_{second}_")
                path = glob.glob(f"{self.db_name}/DATA/{year}_{month}_{day}.{hour}_{minute}_*")[0]
                related_event = obspy.read_events(f"{path}/event.xml")
                event_time = related_event[0].origins[0].time
                event_lat = related_event[0].origins[0].latitude
                event_lon = related_event[0].origins[0].longitude
                event_depth = related_event[0].origins[0].depth
                event_mag = related_event[0].magnitudes[0].mag
                sta_lat = self.inventory.select(network = network, station = station)[0][0].latitude
                sta_lon = self.inventory.select(network = network, station = station)[0][0].longitude
                tr.stats.sac = {'stla':sta_lat, 'stlo':sta_lon, 'evla': event_lat, 'evlo': event_lon, 'evdp': event_depth, 'mag': event_mag, 'o': 0, 'kstnm': station, 'kcmpnm': channel, 'knetwk': network, 'khole': location, 'nzyear': year, 'nzjday': event_time.julday, 'nzhour': hour, 'nzmin': minute, 'nzsec': second, 'nzmsec': event_time.microsecond/1000}
                tr.write(f"{path}/raw.{tr.stats.channel}.{tr.stats.location}.sac", format="SAC")
            except:
                print("Error! there is no directory for this waveform. --> ", f"{self.db_name}/DATA/{year}_{month}_{day}.{hour}_{minute}_{second}_")
        

########################################################################################################################################################################
    def get_waveform_nrcan(self, trace_info, path, notific):
        """
        This function is written for getting single 3-C-waveform from NRCAN and writing them in a sac file.
        """
        network = trace_info["network"]
        station = trace_info["station"]
        channel = trace_info["channel"]
        location = trace_info["location"]
        year = trace_info["year"]
        month = trace_info["month"]
        day = trace_info["day"]
        hour = trace_info["hour"]
        minute = trace_info["minute"]
        second = trace_info["second"]
        date = UTCDateTime(year, month, day, hour, minute, second)
        event = obspy.read_events(f"{path}/event.xml")
        evla = event[0].origins[0].latitude
        evlo = event[0].origins[0].longitude
        evdp = event[0].origins[0].depth
        mag = event[0].magnitudes[0].mag
        stla = self.inventory.select(network = network, station = station)[0][0].latitude
        stlo = self.inventory.select(network = network, station = station)[0][0].longitude
        starttime = f"{year}-{month}-{day}"
        endtime = f"{year}-{month}-{int(day)+1}"
        header = {"evla": evla, "evlo": evlo, "evdp": evdp, "mag": mag, "stla": stla, "stlo": stlo}
        with open(f"{path}/header.txt", "wb") as f:
            pickle.dump(header, f)

        command = f"https://www.earthquakescanada.nrcan.gc.ca/fdsnws/dataselect/1/query?starttime={starttime}&endtime={endtime}&network={network}&station={station}&nodata=404"
        try:
            st = obspy.read(command)
            st.trim(starttime=date, endtime=date + self.siglen)
            st.detrend("demean")
            for tr in st:
                tr.stats.sac = header
                tr.write(f"{path}/raw.{tr.stats.channel}.{tr.stats.location}.nrcan.sac", format="SAC")
        except:
            if notific:
                print(f"Not Found on NRCan: {path} \n",{command},"\n------------------------\n", end='\r')
            else:
                pass
        
        
########################################################################################################################################################################
    def complete_by_nrcan(self, notif = False):
        """
        it first checks which network you want and then select those networks from the inventory and then
        it reads the inventory and the event catalog and then it reads the waveforms and then it writes the waveforms in the sac format.
        """
        open_bulk_file = np.loadtxt(f"{self.db_name}/01_bulk_file.txt", dtype=str)
        bulk = open_bulk_file.tolist()
        
        for waveform in bulk:
            trace_info = {"network":waveform[0],
                          "station":waveform[1],
                          "channel":waveform[3],
                          "location":waveform[2],
                          "year":UTCDateTime(waveform[4]).year,
                          "month":UTCDateTime(waveform[4]).month,
                            "day":UTCDateTime(waveform[4]).day,
                            "hour":UTCDateTime(waveform[4]).hour,
                            "minute":UTCDateTime(waveform[4]).minute,
                            "second":UTCDateTime(waveform[4]).second}
            
            path = f"{self.db_name}/{waveform[0]}_{waveform[1]}/{UTCDateTime(waveform[4]).year}_{UTCDateTime(waveform[4]).month}_{UTCDateTime(waveform[4]).day}.{UTCDateTime(waveform[4]).hour}_{UTCDateTime(waveform[4]).minute}_{UTCDateTime(waveform[4]).second}_"
            if os.path.exists(path):
                if len(os.listdir(path)) < 4:
                    if notif:
                        command = self.get_waveform_nrcan(trace_info, path, notific = True)
                    else:
                        command = self.get_waveform_nrcan(trace_info, path, notific = False)


########################################################################################################################################################################
    def cleaning_empty_directories(self, confirmation = False, sound_on=True):
        """
        it removes the directories that have less than 3 sac files in them.

            inputs:
                main_dir: the main directory that contains the directories that you want to clean
                log_on: if it is True then it will save a txt file in the main directory that contains the directories that have been removed and the directories that have been kept
                sound_on: if it is True then it will play a sound when the cleaning is done

            outputs:
                it doesn't return anything but it removes the directories that have less than 3 sac files in them.
        """
        ask_permission = "y"
        if confirmation:
            ask_permission = input("Do you want to clean the empty directories? (y/n)")

        if ask_permission == "y":
            num_deleted = 0
            for dir in self.paths:
                if dir[1] == "NOT":
                    shutil.rmtree(dir[0])
                    num_deleted += 1
            print(f"Important Messege! {num_deleted} directories have been deleted.")
            root_folder = [folder for folder in os.listdir(self.db_name) if "." not in folder]
            for root in root_folder:
                if len(os.listdir(f"{self.db_name}/{root}")) == 0:
                    shutil.rmtree(f"{self.db_name}/{root}")
        else:
            print("Nothing has been deleted.")
                



########################################################################################################################################################################
    def report(self):
    
        all_path = []
        path = [folder for folder in os.listdir(self.db_name) if "." not in folder]
        for mother in path:
            sub_path = [f"{self.db_name}/{mother}/{folder}" for folder in os.listdir(f"{self.db_name}/{mother}") if folder.endswith("_")]
            for sub in sub_path:
                if len(os.listdir(sub)) > 3:
                    all_path.append([sub, "OK"])
                else:
                    all_path.append([sub, "NOT"])
        self.paths = all_path   
        np.savetxt(f"{self.db_name}/01_paths.txt", self.paths, fmt="%s")        



############################################################################################################################################################################
# Class for making RFs
############################################################################################################################################################################
class make_rf:
############################################################################################################################################################################
    def __init__(self, inputs):
        self.db_name = inputs["db_name"]
        self.path_file_name = inputs["path_file_name"]
        self.paths = open(f"{self.db_name}/{self.path_file_name}", "r").readlines()
        self.rotate_aligment = inputs["rotate_aligment"]
        self.inventory = obspy.read_inventory(f"{self.db_name}/{inputs['inventory_path']}", format="STATIONXML")
        self.efferctive_paths()


############################################################################################################################################################################
    def efferctive_paths(self):
        """
        This function is used to create a list of paths that are not created yet.
        """
        paths = []
        for path in self.paths:
            address_line = path.split("\n")[0].split(" ")[0]
            if path.split("\n")[0].split(" ")[1] == "OK":
                paths.append(address_line)
        self.paths = paths

        
############################################################################################################################################################################
    def make_rf_by_rfpy(self):
        """
        There are two options:
        1. if the file 'all_paths_full.txt' exists, we will use it.
        2. if the file 'all_paths_full.txt' does not exist, we will use the file 'all_paths.txt' and we will create the file 'all_paths_full.txt'
        """

        #defning the taup model
        model = TauPyModel(model="iasp91")

        print("Important Messege! The SNR value is being stored in user0 of sac headers.")
        
        #craeting a dictionary of stations
        stations = self.stations_to_list()   # converting the inventory to a dict of stations

        #defining a new reference file
        detail = [] #[path, snr_Z, snr_R, rf_status, "P_arrival", "RF_QC_comments", "P_phase"]
        #looping over all the paths
        print("If there are already RFs for an event, they will be deleted and new RFs will be created.")
        for path in self.paths:
            path_to_folder = path.split("\n")[0].split(" ")[0]
            # print(f"Working on {path_to_folder}")
            net, sta = self.net_and_sta_from_path(path_to_folder)   #extracting the name of the station and the network from the path string
            sta = stations[f"{net}_{sta}"]
            if exists(f"{path_to_folder}/RFZ.sac"):
                # print(f"RF file(s) already exist(s) for {path_to_folder}. Deleted and new RFs will be created.")
                os.remove(f"{path_to_folder}/RFZ.sac")
                os.remove(f"{path_to_folder}/RFR.sac")
                os.remove(f"{path_to_folder}/RFT.sac")
                    
        #reading the data and the event
            st = obspy.read(f"{path_to_folder}/raw*.sac")
            event = obspy.read_events(f"{path_to_folder}/event.xml")[0]
            
        #extracting P arrival time using TauPy
            depth = 0 if np.isnan(st[0].stats.sac.evdp) else st[0].stats.sac.evdp/1000
            # print("Depth:", depth, "km", path_to_folder)
            P_phase = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=st[0].stats.sac.gcarc, phase_list=["P"])
        
        #checking if there is a P phase and then initializing the RFData object
            if len(P_phase) == 0:
                # print(f"Failed to get P phase for {path_to_folder}")
                # print("It is maybe due to the fact that for this distance there is no P-phase.\nSkipping this station...")
                continue
            else:
                P_phase = event.origins[0].time + P_phase[0].time     #estimated P arrival time using TAUPy
                st.filter("bandpass", freqmin=0.05, freqmax=0.5)
                st.trim(P_phase -60, P_phase + 120)  
                st.detrend("demean")                                   
            
            #creating the RFData object
                rfdata = RFData(sta)
                rfdata.add_event(event)
                rfdata.add_data(st)

            try:
                try:
                    rfdata.rotate(align="ZRT")
                except:
                    # print(f"failed to rotate {path_to_folder}")
                    # print("the problem may be having not the same time span, so we will try to cut the data...")
                    st = self.set_time_span(st)
                    rfdata.data = st
                    rfdata.rotate(align=self.rotate_aligment)

                try:
                    rfdata.calc_snr()
                    snr_Z, snr_R = rfdata.meta.snr, rfdata.meta.snrh
                    rfdata.deconvolve(method="water", wlevel=0.001)
                    rfstream = rfdata.to_stream()

                    info_dict = rfdata.meta.__dict__                
                    with open(f"{path_to_folder}/info_dic.pkl", "wb") as f:
                        pickle.dump(info_dict, f)
                    
                except:
                    # print(f"failed to calculate SNR for {path_to_folder}")
                    snr_Z, snr_R = "None", "None"

                
                for tr in rfstream:
                    tr.stats.sac.user0 = rfdata.meta.snr if rfdata.meta.snr != "None" else "None"
                    tr.write(f"{path_to_folder}/{tr.stats.channel}.sac", format="SAC")
                  
                detail.append([path_to_folder, snr_Z, snr_R, P_phase ])
                # print([path_to_folder, snr_Z, snr_R, P_phase])
                        
            except:                
                detail.append([path_to_folder, "None", "None", "None"])
                if exists(f"{path_to_folder}/RFZ.sac"):
                    print(f"RF file(s) for {path_to_folder} is being deleted.")
                    print("------------------")
                    os.remove(f"{path_to_folder}/RFZ.sac")
                    os.remove(f"{path_to_folder}/RFR.sac")
                    os.remove(f"{path_to_folder}/RFT.sac")
            

        np.savetxt(f"{self.db_name}/02_detail.txt", detail, fmt="%s")


############################################################################################################################################################################
    def net_and_sta_from_path(self, path):
        net = path.split("/")[1].split("_")[0]
        sta = path.split("/")[1].split("_")[1]
        return net, sta


############################################################################################################################################################################
    def stations_to_list(self):
        """
        this function is used to create a list of stations from an inventory file which is essential when using the RFData class.
        """
        stations_table = {}
        for net in self.inventory:
            for sta in net:
                stations_table[f"{net.code}_{sta.code}"] = sta
        return stations_table


############################################################################################################################################################################
    def cleaning_no_rf_directories(self, log_on=False, sound_on=False):
        """
        through this function, we are reading the file 'guide_file_after_RF.txt' and we are deleting all the directories that are not in this file.
        
        Input(s):
        main_dir: the directory where all the directories of the stations are located.
        log_on: if True, the function will print the directories that are deleted.
        sound_on: if True, the function will make a sound when a directory is deleted.
        
        Output(s):
        None
        
        """

        with open(f"{self.db_name}/guide_file_after_RF.txt", "r") as log_file:
            lines = log_file.readlines()
        log_cleaning_after_RF_temp = [["path","snr_value","rf_status","deleted", "comments"]]
        
        for line in lines:
            line = line.replace("\n", "")
            path = line.split(" ")[0]
            snr_value = line.split(" ")[1]
            rf_status = line.split(" ")[2]
            comments = ""
            
            if rf_status == "Not":
                try:
                    shutil.rmtree(path)
                    deleted = True
                except:
                    deleted = None
                    comments = "directory does not exist"
        
            else:
                deleted = False

            if log_on:
                print(f"Deleted:{deleted}\n{path}\n------------------")

            log_cleaning_after_RF_temp.append([path, snr_value, rf_status, deleted, comments])
        
        #saving the log file    
        with open(f"{self.db_name}/guide_file_after_RF_and_cleaning.txt", "w") as log_file:
            for line in log_cleaning_after_RF_temp:
                log_file.write(f"{line}\n")

        if sound_on:
            os.system("say the procedure is done and the log file is saved")


############################################################################################################################################################################
    def set_time_span(self, st):
        t1min = st[0].stats.starttime
        t1max = st[0].stats.endtime
        t2min = st[1].stats.starttime
        t2max = st[1].stats.endtime
        t3min = st[2].stats.starttime
        t3max = st[2].stats.endtime

        t_min = max(t1min, t2min, t3min)
        t_max = min(t1max, t2max, t3max)
        st.trim(t_min, t_max, pad=True, fill_value=0)
        return st


############################################################################################################################################################################
    def get_snr_max(st, seconds=30):
        start_time = st[0].stats.starttime   # start time of 10 seconds after beginning
        end_time = st[0].stats.endtime  # end time of 20 seconds after beginning
        sig_avg_time = (end_time - start_time) / 2
        sig_start = start_time + sig_avg_time - seconds/2
        sig_end = start_time + sig_avg_time + seconds/2
        noise_start = start_time 
        noise_end = start_time + seconds
        signal = st.copy().trim(sig_start, sig_end)
        noise = st.copy().trim(noise_start, noise_end)
        signal_max_Z, signal_max_R, signal_max_T = signal[0].data.max(), signal[1].data.max(), signal[2].data.max()
        noise_max_Z, noise_max_R, noise_max_T = noise[0].data.max(), noise[1].data.max(), noise[2].data.max()
        snr_Z, snr_R, snr_T = 10 * np.log10(signal_max_Z / noise_max_Z), 10 * np.log10(signal_max_R / noise_max_R), 10 * np.log10(signal_max_T / noise_max_T)

        return snr_Z, snr_R, snr_T


############################################################################################################################################################################
    def get_snr_power(st, seconds=30):
        start_time = st[0].stats.starttime   # start time of 10 seconds after beginning
        end_time = st[0].stats.endtime  # end time of 20 seconds after beginning
        sig_avg_time = (end_time - start_time) / 2
        sig_start = start_time + sig_avg_time - seconds/2
        sig_end = start_time + sig_avg_time + seconds/2
        noise_start = start_time 
        noise_end = start_time + seconds
        signal = st.copy().trim(sig_start, sig_end)
        noise = st.copy().trim(noise_start, noise_end)
        signal_pow_Z, signal_pow_R, signal_pow_T = np.sum(signal[0].data**2), np.sum(signal[1].data**2), np.sum(signal[2].data**2)
        noise_pow_Z, noise_pow_R, noise_pow_T = np.sum(noise[0].data**2), np.sum(noise[1].data**2), np.sum(noise[2].data**2)
        snr_Z, snr_R, snr_T = 10 * np.log10(signal_pow_Z / noise_pow_Z), 10 * np.log10(signal_pow_R / noise_pow_R), 10 * np.log10(signal_pow_T / noise_pow_T)

        return snr_Z, snr_R, snr_T

############################################################################################################################################################################
    def rf_report(self):
        with open(f"{self.db_name}/02_detail.txt", "r") as f:
            lines = f.readlines()
            all_sta = {}
            for path in lines:
                path = path.split(" ")[0].split("\n")[0]
                content = os.listdir(path)
                net_sta = path.split("/")[1]
                if len(content) > 6:
                    if net_sta not in all_sta:
                        all_sta[net_sta] = 1
                    else:
                        all_sta[net_sta] += 1

        with open(f"{self.db_name}/01_Stats_{self.db_name}.txt", "w") as f:
            total_events = 0
            Message = "In this version, we created RFs for stations where possible and the next step is doing QC by considering SNR and looking by eyes.\n\n"
            f.write(Message)
            for sta, values in zip(all_sta.keys(), all_sta.values()):
                print(f"{sta} : {values} events")
                f.write(f"{sta} : {values} events\n") 
                total_events += values
            f.write(f"\nThe number of events for all stations are: {total_events}")
            print(f"\nThe number of events for all stations are: {total_events}")


############################################################################################################################################################################

#############################################################################################################################################
# This class is used to do Quality Control over RFs created using make_rf class
#############################################################################################################################################
class rf_qc():
    """
    This class is used to perform quality control on receiver functions. 
    to initiate the class, the user must provide a dictionary of input values.
    input_values_for_qc
    """
#############################################################################################################################################
    def __init__(self, input_values_for_qc):
        self.db_name = input_values_for_qc['db_name']
        self.sorting_qc_list = input_values_for_qc['sorting_qc_list']
        self.hist_bin_number = input_values_for_qc['hist_bin_number']
        self.number_snr_plots = input_values_for_qc['number_snr_plots']
        if self.number_snr_plots%2 != 0:  #to ensure this value is always even
            self.number_snr_plots += 1
        
        self.mk_QC_list()



#############################################################################################################################################
    def mk_QC_list(self):
        path = f"{self.db_name}/02_detail.txt"
        
        if not os.path.isfile(path):
            raise Exception(f"File {path} does not exist. This file should be created using 'make_rf' class.")
        
        with open(path, "r") as f:
            details = f.read().splitlines()
        detail_length = len(details)

        QC_list = []
        for counter, each_line in enumerate(details):
            path = each_line.split(" ")[0]
            if each_line.split(" ")[1] == "None":
                continue
            snr_Z = float(each_line.split(" ")[1])
            snr_R = float(each_line.split(" ")[2])
            P_phase = each_line.split(" ")[3]
            BHorHH = glob.glob(f"{path}/raw*")[0].split("/")[-1].split("_")[0].split(("."))[1][0:2]
            delimiter = "--"
            sta = path.split("/")[1]
            QC_QUAL = 0                                                                   # 0: not QCed, 1-3: QCed based on the quality
            QC_list.append([path, snr_Z, snr_R, sta, delimiter, QC_QUAL, P_phase, BHorHH])           # print(path, snr_Z, snr_R, sta, delimiter, QC_by_eye, P_phase, BHorHH)        
            self.show_percentage_done(counter, detail_length)

        self.QC_list = pd.DataFrame(QC_list, columns=["path", "snr_Z", "snr_R", "NET_STA", "--", "QC_QUAL", "P_time", "BHorHH"])
        


#############################################################################################################################################
    def show_percentage_done(self, for_loop_counter, variable_length):
        # this function is used to show the percentage of the job done in the for loop
        # for_loop_counter: the counter of the for loop
        # variable_length: the length of the variable that is used in the for loop
        # this function is needed to be used inside the for loop which has a counter or enumerate

        print(f"QC list progress: {(for_loop_counter/variable_length*100)//1+1}% is completed.", end="\r")


#############################################################################################################################################
    def snr_hist(self):
        # this function is used to plot the histogram of SNR of the receiver function
        #you can change bin_number to see the histogram with different bin numbers.
        snr_list = self.QC_list[self.sorting_qc_list].tolist()
        plt.hist(snr_list, bins=self.hist_bin_number)
        plt.xlabel("SNR")
        plt.ylabel("Count")
        plt.show()


#############################################################################################################################################
    def visualize_snr(self):
        # this function is used to visualize the SNR of the receiver function to see what would be the cutoff value for SNR
        #we seperate the RFs those have SNR available, but not "None".
        QC_list = self.QC_list[self.QC_list[self.sorting_qc_list] != "None"]

        #selecting the SNR values that we want to plot.
        max_snr = np.max(QC_list[self.sorting_qc_list].astype(float))
        selected_snr = np.linspace(1, max_snr*0.5, self.number_snr_plots)
        fig, axs = plt.subplots(nrows=int(self.number_snr_plots/2), ncols=2, figsize=(15, 14))
        axs = axs.flatten()

        #looping over the selected SNR values.
        counter = 0
        for snr in selected_snr:
            for line in QC_list.values:
                if np.abs(float(line[2])-snr) < 1:
                    
                #plotting the RFs
                    tr = obspy.read(f"{line[0]}/RFR.sac")
                    tr.filter("bandpass", freqmin=0.01, freqmax=0.5)
                    tr.normalize()
                    time = tr[0].times("matplotlib")
                    data = tr[0].data

                #plotting the data       
                    axs[counter].plot(num2date(time), data,linewidth=0.45)
                    axs[counter].xaxis_date()
                    axs[counter].set_title(f"SNR: {round(line[2],0)} /\ {tr[0].stats.sac.kcmpnm}")
                    axs[counter].set_ylabel("Amplitude")
                    axs[counter].set_xlim([time[0], time[-1]])
                                   
                    counter += 1
                    break
        axs[len(selected_snr)-1].set_xlabel("Time (s)") 
        axs[len(selected_snr)-2].set_xlabel("Time (s)") 

        plt.tight_layout()


#############################################################################################################################################
    def snr_threshold_report(self, snr_threshold):
        # this function is used to report the number of RFs that has a SNR value higher than threshold
        # snr_threshold: the threshold value for SNR
        self.snr_threshold = snr_threshold
        
        self.QC_list = self.QC_list[self.QC_list[self.sorting_qc_list] > snr_threshold]
        
        QC_list_grouped = self.QC_list.groupby("NET_STA").count()["path"]

        with open(f"{self.db_name}/03_snr_report.txt", "w") as f:
            f.write(f"SNR threshold: {snr_threshold}\n")
            f.write(f"Total number of events: {QC_list_grouped.sum()}\n")
            f.write(f"Number of events per station:\n")
            f.write(f"{QC_list_grouped}\n")
        self.QC_list.to_csv(f"{self.db_name}/03_QC_list.tmp", index=False)
                    
    
#############################################################################################################################################
    
    def QC_by_eye(self):
        # !!!!!!!! this function is being able to be run individually after intiating the class.
        # this function is used to perform the quality control by eye.
        # the user can go through the RFs and decide if the RF is good or not.      
        try:
            QC_list = pd.read_csv(f"{self.db_name}/03_QC_list.txt")
            print("working on previous QC list...")
        except:
            QC_list = pd.read_csv(f"{self.db_name}/03_QC_list.tmp")
            
        QC_list = self.QC_eye_loop(QC_list)
        QC_list.to_csv(f"{self.db_name}/03_QC_list.txt", index=False)
        print("percentage of remaining:" ,round(len(QC_list[QC_list["QC_QUAL"]==0])/len(QC_list)*100, 1),"%")
            

#############################################################################################################################################      
    def QC_eye_loop(self, QC_list):
        #QC_list_local is the list of RFs that have SNR higher than the threshold value.
        
        for index, values in QC_list.iterrows():
            path = values["path"]
            if values["QC_QUAL"] in ["0", 0]:
                obspy.read(f"{path}/RFR.sac").plot()
                qc_value = input("QC value: (1: low, 2: Medium, 3:high) ---- (press 'q' to exit):")
                if qc_value in ["1", "2", "3"]:
                    QC_list["QC_QUAL"][index] = qc_value
                elif qc_value in ["q", "exit"]:
                    break
                else:
                    QC_list["QC_QUAL"][index] = "1"

        return QC_list
    



    #############################################################################################################################################
    ### Functions
    #############################################################################################################################################

def rf_stats(input_stats_and_viz):
    # a function to see the number of stations in each quality category
    QC_list_file = pd.read_csv(f'{input_stats_and_viz["db_name"]}/03_QC_list.txt')
    QC_list = QC_list_file[["QC_QUAL", "NET_STA", "snr_Z"]].groupby(['QC_QUAL',"NET_STA"]).count()
    QC_list.head()
    return QC_list
    ##############################################################################################################################################
def rf_viz(input_stats_and_viz):
    # a function to visualize the number of stations in each quality category
    QC_list_file = pd.read_csv(f'{input_stats_and_viz["db_name"]}/03_QC_list.txt')
    QC_list_file = QC_list_file[QC_list_file["QC_QUAL"] == input_stats_and_viz["which_quality"]]
    paths = QC_list_file["path"].tolist()
    complete_paths = []
    for path in paths:
        complete_paths.append(f"{path}/RFR.sac")
    st = obspy.Stream()
    for path in complete_paths:
        obspy.read(path).plot()
    np.savetxt(f"{input_stats_and_viz['db_name']}/04_paths_quality_{input_stats_and_viz['which_quality']}.txt", complete_paths, fmt="%s")
    # return st