import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, json
from scipy import stats

np.random.seed(42)
N = 2000
sns.set_theme(style="whitegrid", palette="deep")

# ---------- 1. DATA SIMULATION ----------
modes = np.random.choice(["Road","Rail","Air","Sea"], N, p=[0.55,0.15,0.12,0.18])
regions = np.random.choice(["North","South","East","West","Central"], N, p=[0.2,0.22,0.18,0.2,0.2])
carriers = np.random.choice(["FastTrans","QuickMove","SafeHaul","EcoShip"], N)
dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(np.random.randint(0,365,N), unit="D")
distance = np.where(modes=="Air", np.random.uniform(800,4000,N),
           np.where(modes=="Sea", np.random.uniform(1000,6000,N),
           np.where(modes=="Rail", np.random.uniform(300,2000,N), np.random.uniform(20,1200,N))))
weight = np.random.lognormal(6.2, 0.8, N).round(1)          # kg
volume = (weight/np.random.uniform(120,250,N)).round(2)      # m3
speed = {"Road":55,"Rail":40,"Air":600,"Sea":25}
base_hours = distance/np.vectorize(speed.get)(modes)
handling = {"Road":4,"Rail":12,"Air":8,"Sea":36}
transit = base_hours + np.vectorize(handling.get)(modes)
month = dates.month.values
peak = np.isin(month,[10,11,12]).astype(int)
transit = transit*(1+0.25*peak) + np.random.gamma(2,2,N)
# weather/congestion: heavier delays for Road in East region
transit = transit*np.where((regions=="East")&(modes=="Road"),1.2,1.0)
transit = transit.round(1)
rate = {"Road":0.9,"Rail":0.5,"Air":4.5,"Sea":0.25}
cost = (distance*np.vectorize(rate.get)(modes)*(0.6+weight/1500) + 50 + 120*peak*np.random.uniform(0.5,1.5,N)
        + np.random.normal(0,60,N)).clip(30).round(2)
promised = (base_hours*1.25 + np.vectorize(handling.get)(modes)*1.1 + 6).round(1)
delay = (transit - promised)
on_time = (delay<=0).astype(int)
df = pd.DataFrame(dict(shipment_id=[f"SH{i:05d}" for i in range(N)], ship_date=dates, mode=modes, region=regions,
    carrier=carriers, distance_km=distance.round(1), weight_kg=weight, volume_m3=volume,
    promised_hours=promised, transit_hours=transit, delay_hours=delay.round(1),
    cost_usd=cost, on_time=on_time))
df["cost_per_kg"]=df.cost_usd/df.weight_kg
df["cost_per_km"]=df.cost_usd/df.distance_km
df["month"]=df.ship_date.dt.month
# inject a few missing values & outliers for realism
df.loc[np.random.choice(N,25,replace=False),"weight_kg"]=np.nan
df.loc[np.random.choice(N,5,replace=False),"cost_usd"]*=4
df.to_csv("data/logistics_data.csv",index=False)

out={}
out["shape"]=df.shape
out["missing"]=df.isna().sum()[lambda s:s>0].to_dict()
df["weight_kg"]=df["weight_kg"].fillna(df.groupby("mode")["weight_kg"].transform("median"))
df["cost_per_kg"]=df.cost_usd/df.weight_kg

# ---------- 2. EDA ----------
num=["distance_km","weight_kg","volume_m3","transit_hours","delay_hours","cost_usd"]
desc=df[num].describe().T
desc["median"]=df[num].median(); desc["skew"]=df[num].skew()
desc.round(2).to_csv("outputs/descriptive_stats.csv")
out["desc"]=desc.round(2).reset_index().to_dict("records")
out["on_time_rate"]=round(df.on_time.mean()*100,1)
out["mode_stats"]=df.groupby("mode").agg(shipments=("shipment_id","count"),avg_transit=("transit_hours","mean"),
    avg_cost=("cost_usd","mean"),cpk=("cost_per_kg","median"),on_time=("on_time","mean")).round(2).reset_index().to_dict("records")
corr=df[["distance_km","weight_kg","volume_m3","transit_hours","delay_hours","cost_usd","cost_per_kg"]].corr()
out["corr_dist_cost"]=round(corr.loc["distance_km","cost_usd"],2)
out["corr_dist_transit"]=round(corr.loc["distance_km","transit_hours"],2)
out["corr_wt_cost"]=round(corr.loc["weight_kg","cost_usd"],2)
out["corr_wt_vol"]=round(corr.loc["weight_kg","volume_m3"],2)
out["corr_delay_cost"]=round(corr.loc["delay_hours","cost_usd"],2)
# outliers IQR
q1,q3=df.cost_usd.quantile([.25,.75]); iqr=q3-q1
out["cost_outliers"]=int(((df.cost_usd>q3+1.5*iqr)).sum())
# peak vs off-peak
peak_df=df[df.month.isin([10,11,12])]; off=df[~df.month.isin([10,11,12])]
t,p=stats.ttest_ind(peak_df.delay_hours,off.delay_hours,equal_var=False)
out["peak_delay"]=round(peak_df.delay_hours.mean(),1); out["off_delay"]=round(off.delay_hours.mean(),1)
out["peak_ot"]=round(peak_df.on_time.mean()*100,1); out["off_ot"]=round(off.on_time.mean()*100,1)
out["ttest_p"]=float(p); out["ttest_t"]=round(t,2)
out["peak_cost"]=round(peak_df.cost_usd.mean(),1); out["off_cost"]=round(off.cost_usd.mean(),1)
# region x mode
rm=df[df["mode"]=="Road"].groupby("region").agg(ot=("on_time","mean"),delay=("delay_hours","mean")).round(3)
out["road_region"]=rm.reset_index().to_dict("records")
out["carrier"]=df.groupby("carrier").agg(ot=("on_time","mean"),cost=("cost_usd","mean")).round(3).reset_index().to_dict("records")
# ANOVA carriers
out["anova_p"]=float(stats.f_oneway(*[g.delay_hours for _,g in df.groupby("carrier")]).pvalue)
# regression
sl,ic,r,pv,se=stats.linregress(df.distance_km,df.cost_usd)
out["reg"]=dict(slope=round(sl,3),intercept=round(ic,1),r2=round(r**2,3))

# ---------- 3. VISUALS ----------
def save(n): plt.tight_layout(); plt.savefig(f"figures/{n}.png",dpi=150); plt.close()

# F1 histogram+KDE transit
fig,ax=plt.subplots(figsize=(8,4.5))
sns.histplot(df.transit_hours.clip(upper=df.transit_hours.quantile(.98)),bins=40,kde=True,ax=ax,color="#1f77b4")
ax.axvline(df.transit_hours.mean(),color="red",ls="--",label=f"Mean {df.transit_hours.mean():.1f} h")
ax.axvline(df.transit_hours.median(),color="green",ls="--",label=f"Median {df.transit_hours.median():.1f} h")
ax.set(title="Distribution of Transit Time (clipped at 98th percentile)",xlabel="Transit time (hours)",ylabel="Shipments"); ax.legend(); save("f1_hist")

# F2 boxplot cost by mode (log)
fig,ax=plt.subplots(figsize=(8,4.5))
sns.boxplot(data=df,x="mode",y="cost_usd",order=["Road","Rail","Air","Sea"],ax=ax)
ax.set_yscale("log"); ax.set(title="Transportation Cost by Mode (log scale)",xlabel="Mode",ylabel="Cost (USD, log)"); save("f2_box")

# F3 scatter distance vs cost w/ mode
fig,ax=plt.subplots(figsize=(8,5))
sns.scatterplot(data=df[df.cost_usd<df.cost_usd.quantile(.995)],x="distance_km",y="cost_usd",hue="mode",alpha=.5,s=18,ax=ax)
ax.set(title="Distance vs Transportation Cost by Mode",xlabel="Distance (km)",ylabel="Cost (USD)"); save("f3_scatter")

# F4 heatmap
fig,ax=plt.subplots(figsize=(7.5,6))
sns.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",center=0,ax=ax,square=True)
ax.set_title("Correlation Matrix of Key Variables"); save("f4_corr")

# F5 monthly trend
mon=df.groupby("month").agg(vol=("weight_kg","sum"),ot=("on_time","mean"),ship=("shipment_id","count")).reset_index()
fig,ax=plt.subplots(figsize=(8,4.5))
ax.bar(mon.month,mon.ship,color="#9ecae1",label="Shipments")
ax.set_xlabel("Month"); ax.set_ylabel("Shipments"); ax.set_xticks(range(1,13))
ax2=ax.twinx(); ax2.plot(mon.month,mon.ot*100,color="crimson",marker="o",label="On-time %"); ax2.set_ylabel("On-time rate (%)"); ax2.grid(False)
ax.set_title("Monthly Shipment Volume and On-Time Delivery Rate")
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax.legend(h1+h2,l1+l2,loc="lower left"); save("f5_trend")

# F6 heatmap on-time by region x mode
pv=df.pivot_table(index="region",columns="mode",values="on_time",aggfunc="mean")*100
fig,ax=plt.subplots(figsize=(7,4.5))
sns.heatmap(pv,annot=True,fmt=".0f",cmap="RdYlGn",vmin=40,vmax=100,ax=ax,cbar_kws={"label":"On-time %"})
ax.set_title("On-Time Delivery Rate (%) by Region and Mode"); save("f6_heat")
out["ot_pivot"]=pv.round(1).to_dict()

# F7 carrier bars
cs=df.groupby("carrier").agg(ot=("on_time","mean"),cost=("cost_usd","mean")).reset_index()
fig,axs=plt.subplots(1,2,figsize=(9,4))
sns.barplot(data=cs,x="carrier",y=cs.ot*100,ax=axs[0],color="#2ca02c"); axs[0].set(title="On-Time Rate by Carrier",ylabel="%",ylim=(0,100))
sns.barplot(data=cs,x="carrier",y="cost",ax=axs[1],color="#ff7f0e"); axs[1].set(title="Avg Cost by Carrier",ylabel="USD")
save("f7_carrier")

# F8 mode share pie? use bar of cost-per-kg & delay violin
fig,ax=plt.subplots(figsize=(8,4.5))
sns.violinplot(data=df,x="mode",y="delay_hours",order=["Road","Rail","Air","Sea"],ax=ax,inner="quartile",cut=0)
ax.axhline(0,color="k",ls="--",lw=1); ax.set(title="Delay Distribution by Mode (>0 = late)",xlabel="Mode",ylabel="Delay (hours)"); save("f8_violin")

json.dump(out,open("outputs/stats.json","w"),indent=1,default=str)
print(json.dumps(out,indent=1,default=str))
