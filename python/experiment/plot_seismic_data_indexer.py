import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

df = pd.read_csv('index_all_old_data.csv')

# Convert starttime to datetime
def to_dt(s):
    try:
        return datetime.strptime(str(s), '%Y%m%d.%H%M%S')
    except Exception:
        return None
df['start_dt'] = df['starttime'].apply(to_dt)

# One row per segment of availability
df = df.dropna(subset=['start_dt'])

grouped = df.groupby(['station','channel'])

fig, ax = plt.subplots(figsize=(12, 0.25*len(grouped)))

yticks = []
yticklabels = []
y = 0

for (station, channel), rows in grouped:
    color = 'C0'
    for _, row in rows.iterrows():
        # For continuous, data is 1800s; for triggered, maybe variable, assume e.g. 10s
        duration = 1800 if row['archive_type']=='continuous' else 10
        ax.plot([row['start_dt'], row['start_dt'] + pd.Timedelta(seconds=duration)], [y, y], lw=4, color=color)
    yticklabels.append(f"{station}_{channel}")
    yticks.append(y)
    y += 1

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_xlabel('Time')
ax.set_ylabel('Station_Channel')
ax.set_title('Seismic Data Availability')
fig.tight_layout()
plt.savefig('data_availability.png', dpi=200)
plt.show()