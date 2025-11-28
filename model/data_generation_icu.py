import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import pickle
import datetime
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')
if not os.path.exists("./data/dict"):
    os.makedirs("./data/dict")
if not os.path.exists("./data/csv"):
    os.makedirs("./data/csv")

    
class Generator():
    def __init__(self,cohort_output,if_mort,if_admn,if_los,feat_cond,feat_proc,feat_out,feat_chart,feat_med,impute,include_time=24,bucket=1,predW=12, save_csv=False):
        self.feat_cond,self.feat_proc,self.feat_out,self.feat_chart,self.feat_med = feat_cond,feat_proc,feat_out,feat_chart,feat_med
        self.cohort_output=cohort_output
        self.impute=impute
        self.save_csv=save_csv
        self.data = self.generate_adm()
        print("[ READ COHORT ]")
        
        self.generate_feat()
        print("[ READ ALL FEATURES ]")
        
        if if_mort:
            self.mortality_length(include_time,predW)
            print("[ PROCESSED TIME SERIES TO INDIVIDUAL LOS TIME INTERVAL  ]")
        elif if_admn:
            self.readmission_length(include_time)
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
        elif if_los:
            self.los_length(include_time)
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
        
        self.smooth_meds(bucket)
        print("[ SUCCESSFULLY SAVED DATA DICTIONARIES ]")
    
    def _fix_ids(self, df):
        df['stay_id'] = df['stay_id'].astype('Int64')
        df['itemid']  = df['itemid'].astype('Int64')
        return df

    
    def generate_feat(self):
        if(self.feat_cond):
            print("[ ======READING DIAGNOSIS ]")
            self.generate_cond()
        if(self.feat_proc):
            print("[ ======READING PROCEDURES ]")
            self.generate_proc()
        if(self.feat_out):
            print("[ ======READING OUT EVENTS ]")
            self.generate_out()
        if(self.feat_chart):
            print("[ ======READING CHART EVENTS ]")
            self.generate_chart()
        if(self.feat_med):
            print("[ ======READING MEDICATIONS ]")
            self.generate_meds()

    def generate_adm(self):
        data=pd.read_csv(f"./data/cohort/{self.cohort_output}.csv.gz", compression='gzip', header=0, index_col=None)
        
        '''data['intime'] = pd.to_datetime(data['intime'], errors='coerce')
        data['outtime'] = pd.to_datetime(data['outtime'], errors='coerce')
        data['los']=pd.to_timedelta(data['outtime']-data['intime'],unit='h')
        data['los']=data['los'].astype(str)
        data[['days', 'dummy','hours']] = data['los'].str.split(' ', -1, expand=True)
        data[['hours','min','sec']] = data['hours'].str.split(':', -1, expand=True)
        data['los']=pd.to_numeric(data['days'])*24+pd.to_numeric(data['hours'])
        data=data.drop(columns=['days', 'dummy','hours','min','sec'])
        data=data[data['los']>0]
        data['Age']=data['Age'].astype(int)'''
        
        # Ensure correct datetime parsing
        data['intime'] = pd.to_datetime(data['intime'], errors='coerce')
        data['outtime'] = pd.to_datetime(data['outtime'], errors='coerce')

         # Robust LOS computation in hours
        data['los'] = (data['outtime'] - data['intime']).dt.total_seconds() / 3600.0

        # Remove rows with missing or nonpositive LOS
        data = data[data['los'].notna() & (data['los'] > 0)]

        # Ensure LOS is numeric
        data['los'] = pd.to_numeric(data['los'], errors='coerce')

        # Enforce integer age safely
        if 'Age' in data.columns:
            data['Age'] = pd.to_numeric(data['Age'], errors='coerce').fillna(0).astype(int)
        
        #print(data.head())
        #print(data.shape)
        print(f"[generate_adm] Loaded cohort: {len(data)} stays, {data['los'].notna().sum()} with LOS")
        print(f"[generate_adm] Loaded cohort: {len(data)} stays, {data['los'].isna().sum()} without LOS")
        return data
    
    def generate_cond(self):
        cond=pd.read_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip', header=0, index_col=None)
        
        cond = self._fix_ids(cond)

        cond=cond[cond['stay_id'].isin(self.data['stay_id'])]
        cond_per_adm = cond.groupby('stay_id').size().max()
        self.cond, self.cond_per_adm = cond, cond_per_adm
    
    def generate_proc(self):
        proc=pd.read_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip', header=0, index_col=None)

        proc = self._fix_ids(proc)

        proc=proc[proc['stay_id'].isin(self.data['stay_id'])]
        proc[['start_days', 'dummy','start_hours']] = proc['event_time_from_admit'].str.split(' ', -1, expand=True)
        proc[['start_hours','min','sec']] = proc['start_hours'].str.split(':', -1, expand=True)
        proc['start_time']=pd.to_numeric(proc['start_days'])*24+pd.to_numeric(proc['start_hours'])
        proc=proc.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        proc=proc[proc['start_time']>=0]

        # derive stop_time from 'endtime' (hours from admit)
        if 'endtime' in proc.columns:
            # endtime formatted like "<days> <HH:MM:SS>"
            proc[['end_days', 'dummy2', 'end_hours']] = proc['stop_time_from_admit'].str.split(' ', -1, expand=True)
            proc[['end_hours', 'emin', 'esec']] = proc['end_hours'].str.split(':', -1, expand=True)
            proc['stop_time'] = pd.to_numeric(proc['end_days']) * 24 + pd.to_numeric(proc['end_hours'])
            proc = proc.drop(columns=['end_days', 'dummy2', 'end_hours', 'emin', 'esec'])
        else:
            # Fallback: if no endtime column is present, treat as point event
            proc['stop_time'] = proc['start_time']

        # Attach LOS to clip/guard
        proc = pd.merge(proc, self.data[['stay_id', 'los']], on='stay_id', how='left')

        # Clip stop_time to LOS
        proc.loc[proc['stop_time'] > proc['los'], 'stop_time'] = proc.loc[proc['stop_time'] > proc['los'], 'los']

        # for ventilation rows missing stop_time, assume continues to LOS
        VENT_ITEMIDS = {225792, 225794}
        m = proc['itemid'].isin(VENT_ITEMIDS) & (proc['stop_time'].isna() | (proc['stop_time'] <= proc['start_time']))
        proc.loc[m, 'stop_time'] = proc.loc[m, 'los']

        # Drop any non-positive durations (point events OK if equal? keep only strictly > start)
        proc = proc[proc['stop_time'] > proc['start_time']]

        del proc['los']
        
        '''###Remove where event time is after discharge time
        proc=pd.merge(proc,self.data[['stay_id','los']],on='stay_id',how='left')
        proc['sanity']=proc['los']-proc['start_time']
        proc=proc[proc['sanity']>0]
        del proc['sanity']'''
        
        self.proc=proc
        
    def generate_out(self):
        out=pd.read_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip', header=0, index_col=None)
        
        out = self._fix_ids(out)
        
        out=out[out['stay_id'].isin(self.data['stay_id'])]
        out[['start_days', 'dummy','start_hours']] = out['event_time_from_admit'].str.split(' ', -1, expand=True)
        out[['start_hours','min','sec']] = out['start_hours'].str.split(':', -1, expand=True)
        out['start_time']=pd.to_numeric(out['start_days'])*24+pd.to_numeric(out['start_hours'])
        out=out.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        out=out[out['start_time']>=0]
        
        ###Remove where event time is after discharge time
        out=pd.merge(out,self.data[['stay_id','los']],on='stay_id',how='left')
        out['sanity']=out['los']-out['start_time']
        out=out[out['sanity']>0]
        del out['sanity']
        
        self.out=out
        
        
    def generate_chart(self):
        chunksize = 5000000
        final=pd.DataFrame()
        for chart in tqdm(pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip', header=0, index_col=None,chunksize=chunksize)):
            chart=chart[chart['stay_id'].isin(self.data['stay_id'])]
            
            chart = self._fix_ids(chart)

            chart[['start_days', 'dummy','start_hours']] = chart['event_time_from_admit'].str.split(' ', -1, expand=True)
            chart[['start_hours','min','sec']] = chart['start_hours'].str.split(':', -1, expand=True)
            chart['start_time']=pd.to_numeric(chart['start_days'])*24+pd.to_numeric(chart['start_hours'])
            chart=chart.drop(columns=['start_days', 'dummy','start_hours','min','sec','event_time_from_admit'])
            chart=chart[chart['start_time']>=0]

            ###Remove where event time is after discharge time
            chart=pd.merge(chart,self.data[['stay_id','los']],on='stay_id',how='left')
            chart['sanity']=chart['los']-chart['start_time']
            chart=chart[chart['sanity']>0]
            del chart['sanity']
            del chart['los']
            
            if final.empty:
                final=chart
            else:
                final=final.append(chart, ignore_index=True)
        
        self.chart=final
        
        
        
    def generate_meds(self):
        meds=pd.read_csv("./data/features/preproc_med_icu.csv.gz", compression='gzip', header=0, index_col=None)
        
        meds = self._fix_ids(meds)
        
        meds[['start_days', 'dummy','start_hours']] = meds['start_hours_from_admit'].str.split(' ', -1, expand=True)
        meds[['start_hours','min','sec']] = meds['start_hours'].str.split(':', -1, expand=True)
        meds['start_time']=pd.to_numeric(meds['start_days'])*24+pd.to_numeric(meds['start_hours'])
        meds[['start_days', 'dummy','start_hours']] = meds['stop_hours_from_admit'].str.split(' ', -1, expand=True)
        meds[['start_hours','min','sec']] = meds['start_hours'].str.split(':', -1, expand=True)
        meds['stop_time']=pd.to_numeric(meds['start_days'])*24+pd.to_numeric(meds['start_hours'])
        meds=meds.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        #####Sanity check
        meds['sanity']=meds['stop_time']-meds['start_time']
        meds=meds[meds['sanity']>0]
        del meds['sanity']
        #####Select hadm_id as in main file
        meds=meds[meds['stay_id'].isin(self.data['stay_id'])]
        meds=pd.merge(meds,self.data[['stay_id','los']],on='stay_id',how='left')

        #####Remove where start time is after end of visit
        meds['sanity']=meds['los']-meds['start_time']
        meds=meds[meds['sanity']>0]
        del meds['sanity']
        ####Any stop_time after end of visit is set at end of visit
        meds.loc[meds['stop_time'] > meds['los'],'stop_time']=meds.loc[meds['stop_time'] > meds['los'],'los']
        del meds['los']
        
        meds['rate']=meds['rate'].apply(pd.to_numeric, errors='coerce')
        meds['amount']=meds['amount'].apply(pd.to_numeric, errors='coerce')
        
        self.meds=meds
    
    def mortality_length(self,include_time,predW):
        print("include_time",include_time)
        
        #self.los=include_time
        #self.data=self.data[(self.data['los']>=include_time+predW)]
        
        # Filter out short stays (<12 hours)
        self.data = self.data[self.data['los'] >= predW]

        # Keep all stays regardless of LOS
        self.hids=self.data['stay_id'].unique()
        
        if(self.feat_cond):
            self.cond=self.cond[self.cond['stay_id'].isin(self.data['stay_id'])]
        
        #self.data['los']=include_time

        ####Make equal length input time series and remove data for pred window if needed
        
        ###MEDS
        if(self.feat_med):
            #self.meds=self.meds[self.meds['stay_id'].isin(self.data['stay_id'])]
            #self.meds=self.meds[self.meds['start_time']<=include_time]
            #self.meds.loc[self.meds.stop_time >include_time, 'stop_time']=include_time

            self.meds = self.meds[self.meds['stay_id'].isin(self.data['stay_id'])]
            self.meds = pd.merge(self.meds, self.data[['stay_id','los']], on='stay_id', how='left', suffixes=('_med', '_cohort'))

            # Determine which LOS column exists (depends on whether meds had its own 'los')
            if 'los_cohort' in self.proc.columns:
                los_col = 'los_cohort'
            else:
                los_col = 'los'

            self.meds = self.meds[self.meds['start_time'] <= self.meds[los_col]]
            self.meds.loc[self.meds.stop_time > self.meds['los_col'], 'stop_time'] = self.meds[los_col]
            self.meds.drop(columns=[los_col], inplace=True)
        
        ###PROCS
        if(self.feat_proc):
            #self.proc=self.proc[self.proc['stay_id'].isin(self.data['stay_id'])]
            #self.proc=self.proc[self.proc['start_time']<=include_time]

            self.proc = self.proc[self.proc['stay_id'].isin(self.data['stay_id'])]
            self.proc = pd.merge(self.proc, self.data[['stay_id','los']], on='stay_id', how='left', suffixes=('_proc', '_cohort'))

            # Determine which LOS column exists (depends on whether proc had its own 'los')
            if 'los_cohort' in self.proc.columns:
                los_col = 'los_cohort'
            else:
                los_col = 'los'
            
            self.proc = self.proc[self.proc['start_time'] <= self.proc[los_col]]
            # Drop the temporary cohort LOS column to avoid confusion
            self.proc.drop(columns=[los_col], inplace=True)

        ###OUT
        if(self.feat_out):
            #self.out=self.out[self.out['stay_id'].isin(self.data['stay_id'])]
            #self.out=self.out[self.out['start_time']<=include_time]

            self.out = self.out[self.out['stay_id'].isin(self.data['stay_id'])]
            self.out = pd.merge(self.out, self.data[['stay_id','los']], on='stay_id', how='left', suffixes=('_out', '_cohort'))

            # Determine which LOS column exists (depends on whether out had its own 'los')
            if 'los_cohort' in self.out.columns:
                los_col = 'los_cohort'
            else:
                los_col = 'los'


            self.out = self.out[self.out['start_time'] <= self.out[los_col]]
            self.out.drop(columns=[los_col], inplace=True)

       ###CHART
        if(self.feat_chart):
            #self.chart=self.chart[self.chart['stay_id'].isin(self.data['stay_id'])]
            #self.chart=self.chart[self.chart['start_time']<=include_time]

            self.chart = self.chart[self.chart['stay_id'].isin(self.data['stay_id'])]
            self.chart = pd.merge(self.chart, self.data[['stay_id','los']], on='stay_id', how='left', suffixes=('_chart', '_cohort'))
            
            # Determine which LOS column exists (depends on whether chart had its own 'los')
            if 'los_cohort' in self.chart.columns:
                los_col = 'los_cohort'
            else:
                los_col = 'los'

            self.chart = self.chart[self.chart['start_time'] <= self.chart[los_col]]
            self.chart.drop(columns=[los_col], inplace=True)

        #self.los=include_time
    def los_length(self,include_time):
        print("include_time",include_time)
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time)]
        self.hids=self.data['stay_id'].unique()
        
        if(self.feat_cond):
            self.cond=self.cond[self.cond['stay_id'].isin(self.data['stay_id'])]
        
        self.data['los']=include_time

        ####Make equal length input time series and remove data for pred window if needed
        
        ###MEDS
        if(self.feat_med):
            self.meds=self.meds[self.meds['stay_id'].isin(self.data['stay_id'])]
            self.meds=self.meds[self.meds['start_time']<=include_time]
            self.meds.loc[self.meds.stop_time >include_time, 'stop_time']=include_time
                    
        
        ###PROCS
        if(self.feat_proc):
            self.proc=self.proc[self.proc['stay_id'].isin(self.data['stay_id'])]
            self.proc=self.proc[self.proc['start_time']<=include_time]
            
        ###OUT
        if(self.feat_out):
            self.out=self.out[self.out['stay_id'].isin(self.data['stay_id'])]
            self.out=self.out[self.out['start_time']<=include_time]
            
       ###CHART
        if(self.feat_chart):
            self.chart=self.chart[self.chart['stay_id'].isin(self.data['stay_id'])]
            self.chart=self.chart[self.chart['start_time']<=include_time]
            
    def readmission_length(self,include_time):
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time)]
        self.hids=self.data['stay_id'].unique()
        
        if(self.feat_cond):
            self.cond=self.cond[self.cond['stay_id'].isin(self.data['stay_id'])]
        self.data['select_time']=self.data['los']-include_time
        self.data['los']=include_time

        ####Make equal length input time series and remove data for pred window if needed
        
        ###MEDS
        if(self.feat_med):
            self.meds=self.meds[self.meds['stay_id'].isin(self.data['stay_id'])]
            self.meds=pd.merge(self.meds,self.data[['stay_id','select_time']],on='stay_id',how='left')
            self.meds['stop_time']=self.meds['stop_time']-self.meds['select_time']
            self.meds['start_time']=self.meds['start_time']-self.meds['select_time']
            self.meds=self.meds[self.meds['stop_time']>=0]
            self.meds.loc[self.meds.start_time <0, 'start_time']=0
        
        ###PROCS
        if(self.feat_proc):
            self.proc=self.proc[self.proc['stay_id'].isin(self.data['stay_id'])]
            self.proc=pd.merge(self.proc,self.data[['stay_id','select_time']],on='stay_id',how='left')
            self.proc['start_time']=self.proc['start_time']-self.proc['select_time']
            self.proc=self.proc[self.proc['start_time']>=0]
            
        ###OUT
        if(self.feat_out):
            self.out=self.out[self.out['stay_id'].isin(self.data['stay_id'])]
            self.out=pd.merge(self.out,self.data[['stay_id','select_time']],on='stay_id',how='left')
            self.out['start_time']=self.out['start_time']-self.out['select_time']
            self.out=self.out[self.out['start_time']>=0]
            
       ###CHART
        if(self.feat_chart):
            self.chart=self.chart[self.chart['stay_id'].isin(self.data['stay_id'])]
            self.chart=pd.merge(self.chart,self.data[['stay_id','select_time']],on='stay_id',how='left')
            self.chart['start_time']=self.chart['start_time']-self.chart['select_time']
            self.chart=self.chart[self.chart['start_time']>=0]
        
            
    def smooth_meds(self,bucket):
        """
        Aggregate MEDS, PROC, OUT, and CHART data into equal time buckets per patient,
        preserving each patient's individual LOS.

        Features added:
        - Mean values per bucket for all signals
        - Last known value within each bucket for selected CHART itemids
        - Ventilation tracking: per-bucket ventilation signal and last known status
        """
        final_meds=pd.DataFrame()
        final_proc=pd.DataFrame()
        final_out=pd.DataFrame()
        final_chart=pd.DataFrame()

        # NEW: a dedicated accumulator for ventilation status
        final_proc_vent = pd.DataFrame()

        #itemids for which last known value over 12h is required.
        last_ids = {227443, 227466, 220228, 220615, 225624, 225668, 220739, 223900, 223901, 227582, 223834, 227287,
                    225792, 225794}
        
        # For PROC: ventilation-related itemids
        VENT_ITEMIDS = {225792, 225794}
        
        # Sort each feature table by time
        if(self.feat_med):
            self.meds=self.meds.sort_values(by=['start_time'])
        if(self.feat_proc):
            self.proc=self.proc.sort_values(by=['start_time'])
        if(self.feat_out):
            self.out=self.out.sort_values(by=['start_time'])
        if(self.feat_chart):
            self.chart=self.chart.sort_values(by=['start_time'])

        
        # ---------------------------------------------------------------------
        # Precompute ventilation events for PROC
        # ---------------------------------------------------------------------
        if self.feat_proc:
            vent_proc = self.proc[self.proc["itemid"].isin(VENT_ITEMIDS)][
                ["stay_id", "itemid", "start_time", "stop_time", "subject_id"]
            ].copy()

            # For overlap detection (signal)
            vent_intervals = vent_proc.copy()

            # For last known status
            ev_start = vent_proc[["stay_id", "itemid", "start_time"]].rename(columns={"start_time": "time"})
            ev_start["status_last"] = 1  # ON

            ev_stop = vent_proc[["stay_id", "itemid", "stop_time"]].rename(columns={"stop_time": "time"})
            ev_stop["status_last"] = 0  # OFF

            vent_events = pd.concat([ev_start, ev_stop], ignore_index=True)
        else:
            vent_intervals = pd.DataFrame()
            vent_events = pd.DataFrame()
        
        # ---------------------------------------------------------------------
        
        for hid in tqdm(self.hids):
            stay_los = int(self.data.loc[self.data['stay_id']==hid, 'los'].iloc[0])
            #tqdm.write(f"stay_id={hid}, LOS={stay_los}")
            t=0
            for i in range(0, stay_los, bucket): 
                ###MEDS
                if(self.feat_med):
                    sub_meds=self.meds[(self.meds['stay_id']==hid) & (self.meds['start_time']>=i) & (self.meds['start_time']<i+bucket)].groupby(['stay_id','itemid','orderid']).agg({'stop_time':'max','subject_id':'max','rate':np.nanmean,'amount':np.nanmean})
                    sub_meds=sub_meds.reset_index()
                    sub_meds['start_time']=t
                    sub_meds['stop_time']=sub_meds['stop_time']/bucket
                    if final_meds.empty:
                        final_meds=sub_meds
                    else:
                        final_meds=final_meds.append(sub_meds, ignore_index=True)
                
                ###PROC
                # =============== PROC ===============
                if self.feat_proc:
                    sub_proc = (
                        self.proc[(self.proc['stay_id']==hid) & (self.proc["start_time"] >= i) & (self.proc["start_time"] < i + bucket)]
                        .groupby(["stay_id", "itemid"])
                        .agg({"subject_id": "max"})
                        .reset_index()
                    )
                    sub_proc["start_time"] = t
                    final_proc = pd.concat([final_proc, sub_proc], ignore_index=True)

                    # --- Ventilation tracking ---
                    _sig = vent_intervals[
                        (vent_intervals["stay_id"] == hid) &
                        (vent_intervals["start_time"] < i + bucket) &
                        (vent_intervals["stop_time"] > i)
                    ]
                    if _sig.empty:
                        sub_sig = pd.DataFrame(columns=["stay_id", "itemid", "signal"])
                    else:
                        sub_sig = (
                            _sig.groupby(["stay_id", "itemid"])
                            .agg(signal=("subject_id", "max"))
                            .reset_index()
                        )
                        sub_sig["signal"] = 1  # any overlap → ventilated

                    # E = i + bucket
                    # _ev = vent_events[vent_events["time"] <= E]

                    # Select only events *inside this bucket*
                    ev_window = vent_events[
                        (vent_events["stay_id"] == hid) &
                        (vent_events["time"] >= i) &
                        (vent_events["time"] < i + bucket)
                    ]

                    #if _ev.empty:
                    #    sub_last = pd.DataFrame(columns=["stay_id", "itemid", "status_last", "last_time"])
                    #else:
                    #    idx_last = _ev.groupby(["stay_id", "itemid"])["time"].idxmax()
                    #    sub_last = _ev.loc[idx_last, ["stay_id", "itemid", "status_last", "time"]].rename(columns={"time": "last_time"})

                    if ev_window.empty:
                        # No vent event inside this bucket — last-known is NaN
                        sub_pv = pd.DataFrame(columns=["stay_id", "itemid", "val_last", "last_time"])
                    else:
                        # Find last event *inside this bucket*
                        idx_last = ev_window.groupby(["stay_id", "itemid"])["time"].idxmax()
                        sub_pv = ev_window.loc[idx_last, ["stay_id", "itemid", "status_last", "time"]]
                        sub_pv = sub_pv.rename(columns={"status_last": "val_last", "time": "last_time"})


                    sub_proc_vent = pd.merge(sub_sig, sub_pv, on=["stay_id", "itemid"], how="outer")
                    sub_proc_vent["start_time"] = t

                    final_proc_vent = pd.concat([final_proc_vent, sub_proc_vent], ignore_index=True)

                        
                ###OUT
                if(self.feat_out):
                    sub_out=self.out[(self.out['stay_id']==hid) & (self.out['start_time']>=i) & (self.out['start_time']<i+bucket)].groupby(['stay_id','itemid']).agg({'subject_id':'max'})
                    sub_out=sub_out.reset_index()
                    sub_out['start_time']=t
                    if final_out.empty:
                        final_out=sub_out
                    else:    
                        final_out=final_out.append(sub_out, ignore_index=True)
                        
                        
                ###CHART
                '''if(self.feat_chart):
                    sub_chart=self.chart[(self.chart['start_time']>=i) & (self.chart['start_time']<i+bucket)].groupby(['stay_id','itemid']).agg({'valuenum':np.nanmean})
                    sub_chart=sub_chart.reset_index()
                    sub_chart['start_time']=t
                    if final_chart.empty:
                        final_chart=sub_chart
                    else:    
                        final_chart=final_chart.append(sub_chart)'''
                
                '''if self.feat_chart:
                    _win = self.chart[
                        (self.chart["stay_id"] == hid) &
                        (self.chart["start_time"] >= i) &
                        (self.chart["start_time"] < i + bucket)
                    ]

                    # Mean within bucket
                    sub_chart_mean = (
                        _win.groupby(["stay_id", "itemid"])
                        .agg({"valuenum": np.nanmean})
                        .reset_index()
                    )

                    # Last value within bucket by latest start_time for selected itemids
                    _win_sorted = _win.sort_values(["stay_id", "itemid", "start_time"])
                    _idx_last = _win_sorted.groupby(["stay_id", "itemid"])["start_time"].idxmax()
                    sub_chart_last = (
                        _win_sorted.loc[_idx_last, ["stay_id", "itemid", "valuenum", "start_time"]]
                        .rename(columns={"valuenum": "valuenum_last", "start_time": "last_time"})
                    )
                    sub_chart_last = sub_chart_last[sub_chart_last["itemid"].isin(last_ids)]

                    # Merge mean and last
                    sub_chart = sub_chart_mean.merge(sub_chart_last, on=["stay_id", "itemid"], how="left")
                    sub_chart["start_time"] = t
                    final_chart = pd.concat([final_chart, sub_chart], ignore_index=True)'''
                
                # Modified CHART processing to include mean and last known for numerical variables, and last_known for categorical modes
                if self.feat_chart:
                    _win = self.chart[
                        (self.chart["stay_id"] == hid) &
                        (self.chart["start_time"] >= i) &
                        (self.chart["start_time"] < i + bucket)
                    ]

                    categorical_modes = {223849, 229314, 227577}

                    _win_num = _win[~_win['itemid'].isin(categorical_modes)]
                    _win_cat = _win[_win['itemid'].isin(categorical_modes)]

                    # Mean within bucket
                    sub_chart_mean = (
                        _win_num.groupby(["stay_id", "itemid"])
                        .agg({"valuenum": np.nanmean})
                        .reset_index()
                    )

                    # Last value within bucket by latest start_time for selected itemids
                    _win_sorted = _win_num.sort_values(["stay_id", "itemid", "start_time"])
                    _idx_last = _win_sorted.groupby(["stay_id", "itemid"])["start_time"].idxmax()
                    sub_chart_last = (
                        _win_sorted.loc[_idx_last, ["stay_id", "itemid", "valuenum", "start_time"]]
                        .rename(columns={"valuenum": "valuenum_last", "start_time": "last_time"})
                    )
                    sub_chart_last = sub_chart_last[sub_chart_last["itemid"].isin(last_ids)]

                    # ---- process categorical chart itemids ----
                    # LAST category + timestamp per bucket
                    if not _win_cat.empty:
                        # find last entry inside this bucket
                        _win_cat_sorted = _win_cat.sort_values(["stay_id", "itemid", "start_time"])
                        idx_last_cat = _win_cat_sorted.groupby(["stay_id", "itemid"])["start_time"].idxmax()

                        sub_chart_cat = (
                            _win_cat_sorted.loc[idx_last_cat, ["stay_id", "itemid", "value", "start_time"]]
                            .rename(columns={"value": "cat_last", "start_time": "cat_last_time"})
                        )
                    else:
                        sub_chart_cat = pd.DataFrame(columns=["stay_id","itemid","cat_last","cat_last_time"])
                    
                    #sub_chart = sub_chart_mean.merge(sub_chart_last, on=["stay_id","itemid"], how="left")
                    #sub_chart = sub_chart.merge(sub_chart_cat, on=["stay_id","itemid"], how="left")
                    #sub_chart["start_time"] = t

                    # numeric part
                    sub_chart_num = sub_chart_mean.merge(sub_chart_last, on=["stay_id","itemid"], how="left")

                    # categorical part (no merge with numeric!)
                    sub_chart_cat["valuenum"] = np.nan   # numeric fields missing
                    sub_chart_cat["valuenum_last"] = np.nan
                    sub_chart_cat["last_time"] = np.nan

                    # combine both
                    sub_chart = pd.concat([sub_chart_num, sub_chart_cat], ignore_index=True)

                    sub_chart["start_time"] = t


                    final_chart = pd.concat([final_chart, sub_chart], ignore_index=True)
                
                t=t+1
            #print("bucket",bucket)
            
            
            
        # Summary statistics
        if self.feat_med and not final_meds.empty:
            f2_meds = final_meds.groupby(["stay_id", "itemid", "orderid"]).size()
            self.med_per_adm = f2_meds.groupby("stay_id").sum().reset_index()[0].max()
            self.medlength_per_adm = final_meds.groupby("stay_id").size().max()

        if self.feat_proc and not final_proc.empty:
            f2_proc = final_proc.groupby(["stay_id", "itemid"]).size()
            self.proc_per_adm = f2_proc.groupby("stay_id").sum().reset_index()[0].max()
            self.proclength_per_adm = final_proc.groupby("stay_id").size().max()

        if self.feat_out and not final_out.empty:
            f2_out = final_out.groupby(["stay_id", "itemid"]).size()
            self.out_per_adm = f2_out.groupby("stay_id").sum().reset_index()[0].max()
            self.outlength_per_adm = final_out.groupby("stay_id").size().max()

        if self.feat_chart and not final_chart.empty:
            f2_chart = final_chart.groupby(["stay_id", "itemid"]).size()
            self.chart_per_adm = f2_chart.groupby("stay_id").sum().reset_index()[0].max()
            self.chartlength_per_adm = final_chart.groupby("stay_id").size().max()

        if not final_proc_vent.empty:
            f2_procvent = final_proc_vent.groupby(["stay_id", "itemid"]).size()
            self.procvent_per_adm = f2_procvent.groupby("stay_id").sum().reset_index()[0].max()
            self.procventlength_per_adm = final_proc_vent.groupby("stay_id").size().max()
        else:
            self.procvent_per_adm = 0
            self.procventlength_per_adm = 0
        
        print("[ PROCESSED TIME SERIES TO INDIVIDUAL LOS TIME INTERVAL ]")
        ###CREATE DICT
#         if(self.feat_chart):
#             self.create_chartDict(final_chart,los)
#         else:
        self.create_Dict(final_meds, final_proc, final_out, final_chart, los=None, final_proc_vent=final_proc_vent)
        
    
    def create_chartDict(self,chart,los):
        dataDic={}
        for hid in self.hids:
            grp=self.data[self.data['stay_id']==hid]
            dataDic[hid]={'Chart':{},'label':int(grp['label'])}
        for hid in tqdm(self.hids):
            ###CHART
            if(self.feat_chart):
                df2=chart[chart['stay_id']==hid]
                val=df2.pivot_table(index='start_time',columns='itemid',values='valuenum')
                df2['val']=1
                df2=df2.pivot_table(index='start_time',columns='itemid',values='val')
                #print(df2.shape)
                add_indices = pd.Index(range(stay_los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2=pd.concat([df2, add_df])
                df2=df2.sort_index()
                df2=df2.fillna(0)
                
                val=pd.concat([val, add_df])
                val=val.sort_index()
                if self.impute=='Mean':
                    val=val.ffill()
                    val=val.bfill()
                    val=val.fillna(val.mean())
                elif self.impute=='Median':
                    val=val.ffill()
                    val=val.bfill()
                    val=val.fillna(val.median())
                val=val.fillna(0)
                
                
                df2[df2>0]=1
                df2[df2<0]=0
                #print(df2.head())
                dataDic[hid]['Chart']['signal']=df2.iloc[:,0:].to_dict(orient="list")
                dataDic[hid]['Chart']['val']=val.iloc[:,0:].to_dict(orient="list")
            
            
                
        ######SAVE DICTIONARIES##############
        with open("./data/dict/metaDic", 'rb') as fp:
            metaDic=pickle.load(fp)
        
        with open("./data/dict/dataChartDic", 'wb') as fp:
            pickle.dump(dataDic, fp)

      
        with open("./data/dict/chartVocab", 'wb') as fp:
            pickle.dump(list(chart['itemid'].unique()), fp)
        self.chart_vocab = chart['itemid'].nunique()
        metaDic['Chart']=self.chart_per_adm
        
            
        with open("./data/dict/metaDic", 'wb') as fp:
            pickle.dump(metaDic, fp)
            
            
    def create_Dict(self, meds, proc, out, chart, los=None, final_proc_vent=None):
        dataDic={}
        #print(los)
        labels_csv=pd.DataFrame(columns=['stay_id','label'])
        labels_csv['stay_id']=pd.Series(self.hids)
        labels_csv['label']=0
#         print("# Unique gender",self.data.gender.nunique())
#         print("# Unique ethnicity",self.data.ethnicity.nunique())
#         print("# Unique insurance",self.data.insurance.nunique())

        for hid in self.hids:
            grp=self.data[self.data['stay_id']==hid]
            stay_los = int(grp['los'].iloc[0])  # individual stay length
            #print('Length of stay:', stay_los)
            dataDic[hid]={'Cond':{},'Proc':{},'Med':{},'Out':{},'Chart':{}, 'ProcVent':{},'ethnicity':grp['ethnicity'].iloc[0],'age':int(grp['Age']),'gender':grp['gender'].iloc[0],'label':int(grp['label']), 'los': stay_los}
            labels_csv.loc[labels_csv['stay_id']==hid,'label']=int(grp['label'])

            # --- NEW: handle variable-length stays ---
            use_variable_los = los is None
            

            #print(static_csv.head())
        for hid in tqdm(self.hids):
            grp=self.data[self.data['stay_id']==hid]
            stay_los = int(grp['los'].iloc[0])  # individual stay length
            demo_csv=grp[['Age','gender','ethnicity','insurance']]
            if self.save_csv:
                if not os.path.exists("./data/csv/"+str(hid)):
                    os.makedirs("./data/csv/"+str(hid))
                demo_csv.to_csv('./data/csv/'+str(hid)+'/demo.csv',index=False)
            
            dyn_csv=pd.DataFrame()
            ###MEDS
            if(self.feat_med):
                feat=meds['itemid'].unique()
                df2=meds[meds['stay_id']==hid]
                df2 = df2[df2['start_time'] < stay_los]
                if df2.shape[0]==0:
                    stay_los = int(grp['los'].iloc[0]) if use_variable_los else los
                    amount=pd.DataFrame(np.zeros([stay_los,len(feat)]),columns=feat)
                    amount=amount.fillna(0)
                    amount.columns=pd.MultiIndex.from_product([["MEDS"], amount.columns])
                else:
                    rate=df2.pivot_table(index='start_time',columns='itemid',values='rate')
                    #print(rate)
                    amount=df2.pivot_table(index='start_time',columns='itemid',values='amount')
                    df2=df2.pivot_table(index='start_time',columns='itemid',values='stop_time')
                    #print(df2.shape)
                    add_indices = pd.Index(range(stay_los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2=pd.concat([df2, add_df])
                    df2=df2.sort_index()
                    df2=df2.ffill()
                    df2=df2.fillna(0)

                    rate=pd.concat([rate, add_df])
                    rate=rate.sort_index()
                    rate=rate.ffill()
                    rate=rate.fillna(-1)

                    amount=pd.concat([amount, add_df])
                    amount=amount.sort_index()
                    amount=amount.ffill()
                    amount=amount.fillna(-1)
                    #print(df2.head())
                    df2.iloc[:,0:]=df2.iloc[:,0:].sub(df2.index,0)
                    df2[df2>0]=1
                    df2[df2<0]=0
                    rate.iloc[:,0:]=df2.iloc[:,0:]*rate.iloc[:,0:]
                    amount.iloc[:,0:]=df2.iloc[:,0:]*amount.iloc[:,0:]
                    #print(df2.head())
                    dataDic[hid]['Med']['signal']=df2.iloc[:,0:].to_dict(orient="list")
                    dataDic[hid]['Med']['rate']=rate.iloc[:,0:].to_dict(orient="list")
                    dataDic[hid]['Med']['amount']=amount.iloc[:,0:].to_dict(orient="list")


                    feat_df=pd.DataFrame(columns=list(set(feat)-set(amount.columns)))
    #                 print(feat)
    #                 print(amount.columns)
    #                 print(amount.head())
                    amount=pd.concat([amount,feat_df],axis=1)

                    amount=amount[feat]
                    amount=amount.fillna(0)
    #                 print(amount.columns)
                    amount.columns=pd.MultiIndex.from_product([["MEDS"], amount.columns])
                
                if(dyn_csv.empty):
                    dyn_csv=amount
                else:
                    dyn_csv=pd.concat([dyn_csv,amount],axis=1)
                
                
                
            
            
            ###PROCS
            if(self.feat_proc):
                feat=proc['itemid'].unique()
                df2=proc[proc['stay_id']==hid]
                df2 = df2[df2['start_time'] < stay_los]
                if df2.shape[0]==0:
                    stay_los = int(grp['los'].iloc[0]) if use_variable_los else los
                    df2=pd.DataFrame(np.zeros([stay_los,len(feat)]),columns=feat)
                    df2=df2.fillna(0)
                    df2.columns=pd.MultiIndex.from_product([["PROC"], df2.columns])
                else:
                    df2['val']=1
                    #print(df2)
                    df2=df2.pivot_table(index='start_time',columns='itemid',values='val')
                    #print(df2.shape)
                    add_indices = pd.Index(range(stay_los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2=pd.concat([df2, add_df])
                    df2=df2.sort_index()
                    df2=df2.fillna(0)
                    df2[df2>0]=1 #Ensures all values are binary
                    #print(df2.head())
                    dataDic[hid]['Proc']=df2.to_dict(orient="list")


                    feat_df=pd.DataFrame(columns=list(set(feat)-set(df2.columns)))
                    df2=pd.concat([df2,feat_df],axis=1)

                    df2=df2[feat]
                    df2=df2.fillna(0)
                    df2.columns=pd.MultiIndex.from_product([["PROC"], df2.columns])
                
                if(dyn_csv.empty):
                    dyn_csv=df2
                else:
                    dyn_csv=pd.concat([dyn_csv,df2],axis=1)
                
                
                
                   
            ###OUT
            if(self.feat_out):
                feat=out['itemid'].unique()
                df2=out[out['stay_id']==hid]
                df2 = df2[df2['start_time'] < stay_los]
                if df2.shape[0]==0:
                    stay_los = int(grp['los'].iloc[0]) if use_variable_los else los
                    df2=pd.DataFrame(np.zeros([stay_los,len(feat)]),columns=feat)
                    df2=df2.fillna(0)
                    df2.columns=pd.MultiIndex.from_product([["OUT"], df2.columns])
                else:
                    df2['val']=1
                    df2=df2.pivot_table(index='start_time',columns='itemid',values='val')
                    #print(df2.shape)
                    add_indices = pd.Index(range(stay_los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2=pd.concat([df2, add_df])
                    df2=df2.sort_index()
                    df2=df2.fillna(0)
                    df2[df2>0]=1
                    #print(df2.head())
                    dataDic[hid]['Out']=df2.to_dict(orient="list")

                    feat_df=pd.DataFrame(columns=list(set(feat)-set(df2.columns)))
                    df2=pd.concat([df2,feat_df],axis=1)

                    df2=df2[feat]
                    df2=df2.fillna(0)
                    df2.columns=pd.MultiIndex.from_product([["OUT"], df2.columns])
                
                if(dyn_csv.empty):
                    dyn_csv=df2
                else:
                    dyn_csv=pd.concat([dyn_csv,df2],axis=1)
                
                
            ###CHART (enhanced: mean + last + last_time + signal + last categorical)
            if self.feat_chart:
                categorical_modes = {223849, 229314, 227577}

                feat = chart['itemid'].unique()
                df2 = chart[chart['stay_id'] == hid]
                df2 = df2[df2['start_time'] < stay_los]

                df2_num = df2[~df2['itemid'].isin(categorical_modes)]
                df2_cat = df2[df2['itemid'].isin(categorical_modes)]


                last_ids = {227443, 227466, 220228, 220615, 225624, 225668, 220739, 223900, 223901, 227582, 223834, 227287,
                    225792, 225794}

                if df2.shape[0] == 0:
                    stay_los = int(grp['los'].iloc[0]) if los is None else los
                    val = pd.DataFrame(np.nan, index=range(stay_los), columns=feat)
                    val_last = pd.DataFrame(np.nan, index=range(stay_los), columns=feat)
                    last_time = pd.DataFrame(np.nan, index=range(stay_los), columns=feat)
                    df2_signal = pd.DataFrame(0, index=range(stay_los), columns=feat)
                else:
                    # mean values (valuenum)
                    val = df2_num.pivot_table(index='start_time', columns='itemid', values='valuenum')

                    # signal = presence of measurement in this bucket
                    df2_num_sig = df2_num.copy()
                    df2_num_sig['val'] = 1
                    df2_num_signal = df2_num_sig.pivot_table(index='start_time', columns='itemid', values='val')

                    # last values and last_time (only for selected itemids)
                    df2_num_last = df2_num[df2_num['itemid'].isin(last_ids)]
                    val_last = df2_num_last.pivot_table(index='start_time', columns='itemid', values='valuenum_last')
                    
                    
                    # last_time = df2.pivot_table(index='start_time', columns='itemid', values='last_time')
                    last_time = df2_num_last.pivot_table(index='start_time', columns='itemid', values='last_time')


                    # Align all indices to stay_los
                    add_idx = pd.Index(range(stay_los)).difference(val.index)

                    #add_df = pd.DataFrame(index=add_idx, columns=val.columns)

                    # Each table gets its OWN padding columns
                    add_df_val = pd.DataFrame(index=add_idx, columns=val.columns)
                    add_df_last = pd.DataFrame(index=add_idx, columns=val_last.columns)
                    add_df_time = pd.DataFrame(index=add_idx, columns=last_time.columns)
                    add_df_sig  = pd.DataFrame(index=add_idx, columns=df2_num_signal.columns)

                    add_dfs = [add_df_val, add_df_last, add_df_time, add_df_sig]
                    
                    tables = [val, val_last, last_time, df2_num_signal]
                    # pad each table independently (per-patient LOS)
                    for i, table in enumerate(tables):
                        padded = pd.concat([table, add_dfs[i]]).sort_index()
                        tables[i] = padded.reindex(range(stay_los))
                    # now restore them back to original names
                    val, val_last, last_time, df2_num_signal = tables

                    # Imputation for val (mean), leave val_last/last_time as NaN
                    if self.impute == 'Mean':
                        val = val.ffill().bfill().fillna(val.mean())
                    elif self.impute == 'Median':
                        val = val.ffill().bfill().fillna(val.median())
                    # val = val.fillna(0) # Removed this since it forces mean values to be 0 if NaN. Not sure if it is correct.
                    # signal is not used for now. So, ignoring the functionality of imputation for it.
                    df2_num_signal = df2_num_signal.fillna(0)
                    df2_num_signal[df2_num_signal > 0] = 1


                    # ---- categorical: last-known mode per timestamp ----
                    cat_cols = sorted(df2_cat['itemid'].unique()) if df2_cat.shape[0] > 0 else sorted(categorical_modes)

                    if df2_cat.shape[0] == 0:
                        # No categorical data for this patient
                        cat_last = pd.DataFrame(np.nan, index=range(stay_los), columns=cat_cols)
                        cat_time = pd.DataFrame(np.nan, index=range(stay_los), columns=cat_cols)

                    else:
                        # Compute categorical pivot tables
                        cat_last_tmp = df2_cat.pivot_table(
                            index='start_time',
                            columns='itemid',
                            values='cat_last',
                            aggfunc='last'
                        )

                        cat_time_tmp = df2_cat.pivot_table(
                            index='start_time',
                            columns='itemid',
                            values='cat_last_time',
                            aggfunc='last'
                        )

                        # Identify missing indices for categorical tables
                        add_idx_cat_last = pd.Index(range(stay_los)).difference(cat_last_tmp.index)
                        add_idx_cat_time = pd.Index(range(stay_los)).difference(cat_time_tmp.index)

                        # Padding frames
                        add_df_cat_last = pd.DataFrame(index=add_idx_cat_last, columns=cat_last_tmp.columns)
                        add_df_cat_time = pd.DataFrame(index=add_idx_cat_time, columns=cat_time_tmp.columns)

                        # Apply padding and reindex
                        cat_last = (
                            pd.concat([cat_last_tmp, add_df_cat_last])
                            .sort_index()
                            .reindex(range(stay_los))
                        )

                        cat_time = (
                            pd.concat([cat_time_tmp, add_df_cat_time])
                            .sort_index()
                            .reindex(range(stay_los))
                        )



                # Assign into dataDic
                dataDic[hid]['Chart']['signal'] = df2_num_signal.to_dict(orient="list")
                dataDic[hid]['Chart']['val'] = val.to_dict(orient="list")
                dataDic[hid]['Chart']['val_last'] = val_last.to_dict(orient="list")
                dataDic[hid]['Chart']['last_time'] = last_time.to_dict(orient="list")

                dataDic[hid]['Chart']['cat_last'] = cat_last.to_dict(orient="list")
                dataDic[hid]['Chart']['cat_last_time'] = cat_time.to_dict(orient="list")


                val.columns = pd.MultiIndex.from_product([["CHART"], val.columns])

                # Add to dyn_csv for export
                dyn_csv = pd.concat([dyn_csv, val], axis=1) if not dyn_csv.empty else val

            
            ### PROCVENT (ventilation tracking)
            if self.feat_proc and (final_proc_vent is not None):
                VENT_ITEMIDS = [225792, 225794]
                dv = final_proc_vent[final_proc_vent['stay_id'] == hid]
                dv = dv[dv['start_time'] < stay_los]

                if dv.shape[0] == 0:
                    signal_pv = pd.DataFrame(0, index=range(stay_los), columns=VENT_ITEMIDS)
                    val_last_pv = pd.DataFrame(np.nan, index=range(stay_los), columns=VENT_ITEMIDS)
                    last_time_pv = pd.DataFrame(np.nan, index=range(stay_los), columns=VENT_ITEMIDS)
                else:
                    sig = dv.pivot_table(index='start_time', columns='itemid', values='signal', aggfunc='max')
                    last = dv.pivot_table(index='start_time', columns='itemid', values='val_last', aggfunc='last')
                    ltime = dv.pivot_table(index='start_time', columns='itemid', values='last_time', aggfunc='last')

                    # cols = list(set(VENT_ITEMIDS) | set(sig.columns) | set(last.columns) | set(ltime.columns))
                    # Stable column ordering: first the canonical vent itemids, then any extras sorted
                    candidate_cols = list(sig.columns) + list(last.columns) + list(ltime.columns)
                    unique_cols = list(dict.fromkeys(candidate_cols))   # preserves order, removes duplicates

                    # Ensure VENT_ITEMIDS first if present
                    cols = [cid for cid in VENT_ITEMIDS if cid in unique_cols] + \
                            [cid for cid in unique_cols if cid not in VENT_ITEMIDS]

                    
                    
                    def _align(df, fill): return df.reindex(index=range(stay_los), columns=cols).fillna(fill)
                    signal_pv = _align(sig, 0)
                    val_last_pv = _align(last, np.nan)
                    last_time_pv = _align(ltime, np.nan)

                # Store into dictionary
                dataDic[hid]['ProcVent'] = {
                    'signal': signal_pv.to_dict('list'),
                    'val_last': val_last_pv.to_dict('list'),
                    'last_time': last_time_pv.to_dict('list')
                }

                # Add to dyn_csv for CSV saving
                val_last_pv.columns = pd.MultiIndex.from_product([["PROCVENT"], val_last_pv.columns])
                dyn_csv = pd.concat([dyn_csv, val_last_pv], axis=1) if not dyn_csv.empty else val_last_pv

            ##########COND#########
            if(self.feat_cond):
                feat=self.cond['new_icd_code'].unique()
                grp=self.cond[self.cond['stay_id']==hid]
                if(grp.shape[0]==0):
                    dataDic[hid]['Cond']={'fids':list(['<PAD>'])}
                    feat_df=pd.DataFrame(np.zeros([1,len(feat)]),columns=feat)
                    grp=feat_df.fillna(0)
                    grp.columns=pd.MultiIndex.from_product([["COND"], grp.columns])
                else:
                    dataDic[hid]['Cond']={'fids':list(grp['new_icd_code'])}
                    grp['val']=1
                    grp=grp.drop_duplicates()
                    grp=grp.pivot(index='stay_id',columns='new_icd_code',values='val').reset_index(drop=True)
                    feat_df=pd.DataFrame(columns=list(set(feat)-set(grp.columns)))
                    grp=pd.concat([grp,feat_df],axis=1)
                    grp=grp.fillna(0)
                    grp=grp[feat]
                    grp.columns=pd.MultiIndex.from_product([["COND"], grp.columns])
            
            if self.save_csv:
                grp.to_csv('./data/csv/'+str(hid)+'/static.csv',index=False)   
                labels_csv.to_csv('./data/csv/labels.csv',index=False)    
            
                
        ######SAVE DICTIONARIES##############
        metaDic={'Cond':{},'Proc':{},'Med':{},'Out':{},'Chart':{},'LOS':{}}
        
        #metaDic['LOS']=los
        metaDic['LOS'] = {hid: dataDic[hid]['los'] for hid in dataDic}
        
        with open("./data/dict/dataDic", 'wb') as fp:
            pickle.dump(dataDic, fp)

        with open("./data/dict/hadmDic", 'wb') as fp:
            pickle.dump(self.hids, fp)
        
        with open("./data/dict/ethVocab", 'wb') as fp:
            pickle.dump(list(self.data['ethnicity'].unique()), fp)
            self.eth_vocab = self.data['ethnicity'].nunique()
            
        with open("./data/dict/ageVocab", 'wb') as fp:
            pickle.dump(list(self.data['Age'].unique()), fp)
            self.age_vocab = self.data['Age'].nunique()
            
        with open("./data/dict/insVocab", 'wb') as fp:
            pickle.dump(list(self.data['insurance'].unique()), fp)
            self.ins_vocab = self.data['insurance'].nunique()
            
        if(self.feat_med):
            with open("./data/dict/medVocab", 'wb') as fp:
                pickle.dump(list(meds['itemid'].unique()), fp)
            self.med_vocab = meds['itemid'].nunique()
            metaDic['Med']=self.med_per_adm
            
        if(self.feat_out):
            with open("./data/dict/outVocab", 'wb') as fp:
                pickle.dump(list(out['itemid'].unique()), fp)
            self.out_vocab = out['itemid'].nunique()
            metaDic['Out']=self.out_per_adm
            
        if(self.feat_chart):
            with open("./data/dict/chartVocab", 'wb') as fp:
                pickle.dump(list(chart['itemid'].unique()), fp)
            self.chart_vocab = chart['itemid'].nunique()
            metaDic['Chart']=self.chart_per_adm
        
        if(self.feat_cond):
            with open("./data/dict/condVocab", 'wb') as fp:
                pickle.dump(list(self.cond['new_icd_code'].unique()), fp)
            self.cond_vocab = self.cond['new_icd_code'].nunique()
            metaDic['Cond']=self.cond_per_adm
        
        if(self.feat_proc):    
            with open("./data/dict/procVocab", 'wb') as fp:
                pickle.dump(list(proc['itemid'].unique()), fp)
            self.proc_vocab = proc['itemid'].nunique()
            metaDic['Proc']=self.proc_per_adm
        
        if self.feat_proc and (final_proc_vent is not None):
            metaDic['ProcVent'] = self.procvent_per_adm

        with open("./data/dict/metaDic", 'wb') as fp:
            pickle.dump(metaDic, fp)
            
            
      


