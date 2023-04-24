import streamlit as st
import pandas as pd
import preprocessor,helper
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.io as pio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np # linear algebra
import os
import folium
from folium import plugins
import geopandas as gpd
import branca
import matplotlib
from textwrap import wrap
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy
from scipy import stats

df = pd.read_csv('athlete_events.csv')
region_df = pd.read_csv('noc_regions.csv')

# st.markdown(f"""
#     <style>
#         .reportview-container .main .block-container{{
#             padding-left: {0}rem;
#         }}
#     </style>""",
#     unsafe_allow_html=True,
# )

dfS=df[df['Season']=='Summer']
dfS=dfS.merge(region_df,on='NOC',how='left')
#Removing duplicates:
dfS.drop_duplicates(inplace=True)
#Concatenating horizontally with original data frame
dfS=pd.concat([dfS,pd.get_dummies(dfS['Medal'])],axis=1)
#Remove duplicate rows on basis of team,noc, games, year, season(can be dropped as its summer only), city, sport, event, medal
#Drop rows where all these values are same:
medal_tallyS0=dfS.drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport','Event','Medal'])
medal_tallyS0.loc[medal_tallyS0['NOC'] == 'URS', 'NOC'] = 'RUS'
medal_tallyS0.loc[medal_tallyS0['NOC'] == 'FRG', 'NOC'] = 'GER'
medal_tallyS0.loc[medal_tallyS0['NOC'] == 'GDR', 'NOC'] = 'GER'
#Now, calculating total medals again:
#NOC ke upar group by kar ke sum karna hai saare columns ka
medal_tallyS0=medal_tallyS0.groupby(['NOC','Year']).sum()[['Gold','Silver','Bronze']].sort_values('NOC',ascending=True).reset_index()
medal_tallyS0['Total']=medal_tallyS0['Gold']+medal_tallyS0['Silver']+medal_tallyS0['Bronze']
#NOC ke upar group by kar ke sum karna hai saare columns ka
medal_tallyS1=medal_tallyS0.groupby(['NOC']).sum()[['Gold','Silver','Bronze']].sort_values('NOC',ascending=True).reset_index()
medal_tallyS1['Total']=medal_tallyS1['Gold']+medal_tallyS1['Silver']+medal_tallyS1['Bronze']
topC=medal_tallyS1[medal_tallyS1['Total']>200]['NOC'].unique()
countries=['AFG', 'AHO', 'ALB', 'ALG', 'AND', 'ANG', 'ANT', 'ANZ', 'ARG',
       'ARM', 'ARU', 'ASA', 'AUS', 'AUT', 'AZE', 'BAH', 'BAN', 'BAR',
       'BDI', 'BEL', 'BEN', 'BER', 'BHU', 'BIH', 'BIZ', 'BLR', 'BOH',
       'BOL', 'BOT', 'BRA', 'BRN', 'BRU', 'BUL', 'BUR', 'CAF', 'CAM',
       'CAN', 'CAY', 'CGO', 'CHA', 'CHI', 'CHN', 'CIV', 'CMR', 'COD',
       'COK', 'COL', 'COM', 'CPV', 'CRC', 'CRO', 'CRT', 'CUB', 'CYP',
       'CZE', 'DEN', 'DJI', 'DMA', 'DOM', 'ECU', 'EGY', 'ERI', 'ESA',
       'ESP', 'EST', 'ETH', 'EUN', 'FIJ', 'FIN', 'FRA', 'FRG', 'FSM',
       'GAB', 'GAM', 'GBR', 'GBS', 'GDR', 'GEO', 'GEQ', 'GER', 'GHA',
       'GRE', 'GRN', 'GUA', 'GUI', 'GUM', 'GUY', 'HAI', 'HKG', 'HON',
       'HUN', 'INA', 'IND', 'IOA', 'IRI', 'IRL', 'IRQ', 'ISL', 'ISR',
       'ISV', 'ITA', 'IVB', 'JAM', 'JOR', 'JPN', 'KAZ', 'KEN', 'KGZ',
       'KIR', 'KOR', 'KOS', 'KSA', 'KUW', 'LAO', 'LAT', 'LBA', 'LBR',
       'LCA', 'LES', 'LIB', 'LIE', 'LTU', 'LUX', 'MAD', 'MAL', 'MAR',
       'MAS', 'MAW', 'MDA', 'MDV', 'MEX', 'MGL', 'MHL', 'MKD', 'MLI',
       'MLT', 'MNE', 'MON', 'MOZ', 'MRI', 'MTN', 'MYA', 'NAM', 'NBO',
       'NCA', 'NED', 'NEP', 'NFL', 'NGR', 'NIG', 'NOR', 'NRU', 'NZL',
       'OMA', 'PAK', 'PAN', 'PAR', 'PER', 'PHI', 'PLE', 'PLW', 'PNG',
       'POL', 'POR', 'PRK', 'PUR', 'QAT', 'RHO', 'ROT', 'ROU', 'RSA',
       'RUS', 'RWA', 'SAA', 'SAM', 'SCG', 'SEN', 'SEY', 'SGP', 'SKN',
       'SLE', 'SLO', 'SMR', 'SOL', 'SOM', 'SRB', 'SRI', 'SSD', 'STP',
       'SUD', 'SUI', 'SUR', 'SVK', 'SWE', 'SWZ', 'SYR', 'TAN', 'TCH',
       'TGA', 'THA', 'TJK', 'TKM', 'TLS', 'TOG', 'TPE', 'TTO', 'TUN',
       'TUR', 'TUV', 'UAE', 'UAR', 'UGA', 'UKR', 'UNK', 'URS', 'URU',
       'USA', 'UZB', 'VAN', 'VEN', 'VIE', 'VIN', 'VNM', 'WIF', 'YAR',
       'YEM', 'YMD', 'YUG', 'ZAM', 'ZIM']
countries_dict = {index:country for index, country in enumerate(countries)}
countries_dict2 = {country:index for index, country in enumerate(countries)}
yrs=medal_tallyS0['Year'].unique()
yrs.sort()
dfNEW=pd.DataFrame({'Year': yrs})
zero13=np.zeros(29)
for i in range(230):
    dfNEW[countries_dict[i]]=zero13
    
zero230=np.zeros(230)
k=0
for yr in yrs:
    t1=medal_tallyS0[medal_tallyS0['Year']==yr].sort_values('NOC',ascending=True).reset_index()
    for i in range(t1.shape[0]):
        zero230[countries_dict2[t1['NOC'].iloc[i]]]+=t1['Total'].iloc[i]
    for j in range(230):
        dfNEW[countries_dict[j]].iloc[k]+=zero230[j]
    k+=1
dfNEW2=dfNEW[topC]
dfNEW2['Year']=yrs


# Melt the dataframe to create a "long" format
df_melted = pd.melt(dfNEW2, id_vars='Year', var_name='variable', value_name='value')
df_oa=pd.read_csv('Players_Medal_list_Olympics.csv')
df_oa=df_oa.groupby(['region','Sport_Type']).agg({'Gold': 'sum', 'Silver': 'sum','Bronze':'sum'}).reset_index()
df_oa['Total']=df_oa['Gold']+df_oa['Silver']+df_oa['Bronze']
# Count the number of occurrences of each region
region_counts = df_oa['region'].value_counts()

# Get the regions that occur exactly 2 times
valid_regions = region_counts[region_counts == 2].index

# Filter the DataFrame to keep only rows with valid regions
df_filtered = df_oa[df_oa['region'].isin(valid_regions)]
# create a mapping between country codes and full country names
country_codes = pd.read_csv('https://raw.githubusercontent.com/datasets/country-codes/master/data/country-codes.csv')

country_map = dict(zip(country_codes['ISO3166-1-Alpha-3'], country_codes['CLDR display name']))
# read the data
medal_tallyS=pd.read_csv('mdl.csv')
# add a new column to the medal_tallyS dataframe with the full country names
medal_tallyS['Country'] = medal_tallyS['NOC'].map(country_map)
cntyL=['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra',
       'Angola', 'Antigua', 'Argentina', 'Armenia', 'Aruba', 'Australia',
       'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh',
       'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda',
       'Bhutan', 'Boliva', 'Bosnia and Herzegovina', 'Botswana', 'Brazil',
       'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia',
       'Cameroon', 'Canada', 'Cape Verde', 'Cayman Islands',
       'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia',
       'Comoros', 'Cook Islands', 'Costa Rica', 'Croatia', 'Cuba',
       'Curacao', 'Cyprus', 'Czech Republic',
       'Democratic Republic of the Congo', 'Denmark', 'Djibouti',
       'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt',
       'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia',
       'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia',
       'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guam',
       'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti',
       'Honduras', 'Hungary', 'Iceland', 'India',
       'Individual Olympic Athletes', 'Indonesia', 'Iran', 'Iraq',
       'Ireland', 'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan',
       'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Kosovo', 'Kuwait',
       'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia',
       'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macedonia',
       'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta',
       'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico',
       'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro',
       'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal',
       'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria',
       'North Korea', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestine',
       'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines',
       'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Republic of Congo',
       'Romania', 'Russia', 'Rwanda', 'Saint Kitts', 'Saint Lucia',
       'Saint Vincent', 'Samoa', 'San Marino', 'Sao Tome and Principe',
       'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone',
       'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia',
       'South Africa', 'South Korea', 'South Sudan', 'Spain', 'Sri Lanka',
       'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria',
       'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste',
       'Togo', 'Tonga', 'Trinidad', 'Tunisia', 'Turkey', 'Turkmenistan',
       'UK', 'USA', 'Uganda', 'Ukraine', 'United Arab Emirates',
       'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam',
       'Virgin Islands, British', 'Virgin Islands, US', 'Yemen', 'Zambia',
       'Zimbabwe']
# Load 21 datasets
population = pd.read_csv('2020/population_by_country_2020.csv')
regions = region_df.copy()
df.copy()
dftmp=df.copy()
dfMF=df.copy()
daf = df.copy()
df_21 = pd.read_csv('2020/Tokyo 2021 dataset v3.csv')
df_21MF=df_21.copy()
df_21_full = pd.read_csv('2020/Tokyo 2021 dataset v4.csv')
df_21_fullMF=df_21_full.copy()
population = pd.read_csv('2020/population_by_country_2020.csv')
regions = pd.read_csv('2016-/noc_regions.csv')

daf = pd.read_csv('2016-/athlete_events.csv')
df_21 = pd.read_csv('2020/Tokyo 2021 dataset v3.csv')
df_21_full = pd.read_csv('2020/Tokyo 2021 dataset v4.csv')


# For geographic plotting
url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
country_shapes = f'{url}/world-countries.json'

# For geographic plotting
global_polygons = gpd.read_file(country_shapes)
global_polygons.to_file('global_polygons.geojson', driver = 'GeoJSON')

#global_polygons.plot(figsize=(10,5)) we now have a map of the globe

# Tabular
daf = pd.merge(daf,regions,left_on='NOC',right_on='NOC')
daf = daf.query('Season == "Summer"') # Only interested in Summer Olympics for this project

#Replacing the country name with common values
daf.replace('USA', "United States of America", inplace = True)
daf.replace('Tanzania', "United Republic of Tanzania", inplace = True)
daf.replace('Democratic Republic of Congo', "Democratic Republic of the Congo", inplace = True)
daf.replace('Congo', "Republic of the Congo", inplace = True)
daf.replace('Lao', "Laos", inplace = True)
daf.replace('Syrian Arab Republic', "Syria", inplace = True)
daf.replace('Serbia', "Republic of Serbia", inplace = True)
daf.replace('Czechia', "Czech Republic", inplace = True)
daf.replace('UAE', "United Arab Emirates", inplace = True)
daf.replace('UK', "United Kingdom", inplace = True)

population.replace('United States', "United States of America", inplace = True)
population.replace('Czech Republic (Czechia)', "Czech Republic", inplace = True)
population.replace('DR Congo', "Democratic Republic of the Congo", inplace = True)
population.replace('Serbia', "Republic of Serbia", inplace = True)
population.replace('Tanzania', "United Republic of Tanzania", inplace = True)

df_21_full.replace('Great Britain', "United Kingdom", inplace = True)
df_21_full.replace("People's Republic of China", "China", inplace = True)
df_21_full.replace("ROC", "Russia", inplace = True)

df = preprocessor.preprocess(df,region_df)
# Function to map country to city

def host_country(col):
    if col == "Rio de Janeiro":
        return "Brazil"
    elif col == "London":
        return "United Kingdom"
    elif col == "Beijing":
        return  "China"
    elif col == "Athina":
        return  "Greece"
    elif col == "Sydney" or col == "Melbourne":
        return  "Australia"
    elif col == "Atlanta" or col == "Los Angeles" or col == "St. Louis":
        return  "United States of America"
    elif col == "Barcelona":
        return  "Spain"
    elif col == "Seoul":
        return  "South Korea"
    elif col == "Moskva":
        return  "Russia"
    elif col == "Montreal":
        return  "Canada"
    elif col == "Munich" or col == "Berlin":
        return  "Germany"
    elif col == "Mexico City":
        return  "Mexico"
    elif col == "Tokyo":
        return  "Japan"
    elif col == "Roma":
        return  "Italy"
    elif col == "Paris":
        return  "France"
    elif col == "Helsinki":
        return  "Finland"
    elif col == "Amsterdam":
        return  "Netherlands"
    elif col == "Antwerpen":
        return  "Belgium"
    elif col == "Stockholm":
        return  "Sweden"
    else:
        return "Other"


# Applying this function
# print(daf.columns)
# medalst=daf[['Sex','Sport','Medal','NOC','region']]
# # medalst=medalst[medalst['Medal'].notna()]
# # countr_map = dict(zip(country_codes['CLDR display name'],country_codes['ISO3166-1-Alpha-3'] ))
# # medalst['NOC'] = medalst['region'].map(countr_map)
# medalst[['Sex','Sport','Medal','NOC','region']].to_csv('mdl1.csv',index=False)
medalst=pd.read_csv('mdl1.csv',header=0)
daf['Host_Country'] = daf['City'].apply(host_country)

df_new = daf.groupby(['Year','Host_Country','region','Medal'])['Medal'].count().unstack().fillna(0).astype(int).reset_index()
# print(df_new)
df_new['Is_Host'] = np.where(df_new['Host_Country'] == df_new['region'],1,0)
df_new['Total Medals'] = df_new['Bronze'] + df_new['Silver'] + df_new['Gold']


# Preparing to add 2021 data to our historic df

df_21_full_refined = df_21_full[['Team/NOC', "Gold Medal", "Silver Medal", "Bronze Medal"]]
df_21_full_refined['Total Medals'] = df_21_full_refined[["Gold Medal", "Silver Medal", "Bronze Medal"]].sum(axis=1)
df_21_full_refined['Year'] = 2021

df_21_full_refined = df_21_full_refined.rename(columns={'Gold Medal':'Gold', 'Silver Medal':'Silver','Bronze Medal':'Bronze'})

df_21_full_refined['Is_Host'] = np.where(df_21_full_refined['Team/NOC'] == 'Japan',1,0)
df_21_full_refined['Host_Country'] = 'Japan'
df_21_full_refined = df_21_full_refined.rename(columns={'Team/NOC':'region'})

# Adding 2021 data to historic
df_new = df_new.append(df_21_full_refined)

# Removing Russia as many Olympic games were competed in as the Soviet Union, containing several modern day nations

df_new = df_new.query("region != 'Russia' | region != 'ROC'")

merged = pd.merge(dfMF, regions, on='NOC', how='left')
# merged.head()
femaleInOlympics = merged[(merged.Sex == 'F') & (merged.Season == 'Summer')]

maleInOlympics = merged[(merged.Sex == 'M') & (merged.Season == 'Summer')]

# Append the two dataframes
dfMF = pd.concat([maleInOlympics, femaleInOlympics], ignore_index=True)


def wmap0():
    # create a world map graph using Plotly
    fig = px.choropleth(medal_tallyS, locations="NOC", color="Total",
                        hover_name="Country",
                        hover_data=["Gold", "Silver", "Bronze", "Total"],
                        projection="natural earth",
                        color_continuous_scale='plasma')

    # customize the graph layout
    fig.update_layout(title="Summer Olympic Medal Counts by Country",
                      geo=dict(showcountries=True,
                               showcoastlines=True))

    # show the graph using Streamlit
    st.plotly_chart(fig)

def wmap():
    
    sportss=['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']
    # create a world map graph using Plotly
    col1, col2, = st.columns(2)
    sex=col1.selectbox("Select Gender",['Overall','Male','Female'])
    sport=col2.selectbox("Select Sports",['Overall']+sportss)
    print(sport)
    df1=medalst
    if(sex!='Overall'):
        if(sex=='Male'):
            df1=df1.loc[df1['Sex']=='M']
        else:
            df1=df1.loc[df1['Sex']=='F']
            # print(df1)
    if(sport!='Overall'):
        df1=df1.loc[df1['Sport']==sport]
    
    df1=df1[['NOC','Medal']].groupby(['NOC','Medal'])['Medal'].count().unstack()
    df1=df1.fillna(0)
    # df1['NOC']=df1.index
    df1=df1.reset_index('NOC')
    df1['Total']=df1['Gold']+df1['Silver']+df1['Bronze']
    country_codes = pd.read_csv('mdl.csv')
    country_ma = dict(zip(country_codes['NOC'], country_codes['region']))
    df1['Country']=df1['NOC'].map(country_ma)
    print(df1)
    df=pd.read_csv('mdl.csv',header=0)
    # df1=df1[['NOC','Medal']].groupby(['NOC','Medal'])['Medal'].count().unstack()
    # df1=df1.fillna(0)
    # df1['NOC']=df1.index
    # df1['Total']=df1['Gold']+df1['Silver']+df1['Bronze']
    df2=df.loc[~df['NOC'].isin(df1['NOC'].tolist())]
    df2=df2[['NOC','Gold','Silver','Bronze','Total','region']]
    df2.iloc[:,1:5] = 0
    df2.columns=['NOC','Gold','Silver','Bronze','Total','Country']
    df1=pd.concat([df1,df2])
    fig = px.choropleth(df1, locations="NOC", color="Total",
                        hover_name="Country",
                        hover_data=["Gold", "Silver", "Bronze", "Total"],
                        projection="natural earth",
                        color_continuous_scale='plasma')

    # customize the graph layout
    fig.update_layout(title="Summer Olympic Medal Counts by Country",
                      geo=dict(showcountries=True,
                               showcoastlines=True),width=1000,height=700)

    # show the graph using Streamlit
    st.plotly_chart(fig)

def wmap2():
    medal_tallyS = pd.read_csv('mdl.csv')
    tmpS = medal_tallyS.drop('Total', axis=1)

    # melt the dataframe to create a long-form version for plotting
    tmpS_melt = pd.melt(tmpS[0:15], id_vars=['NOC'], value_vars=['Gold', 'Silver', 'Bronze'], var_name='Medal', value_name='Count')
    
    # create the stacked bar chart using plotly graph objects
    fig = go.Figure()
    
    # Add a trace for Gold Medals
    fig.add_trace(go.Bar(
        x=tmpS_melt[tmpS_melt['Medal'] == 'Gold']['NOC'],
        y=tmpS_melt[tmpS_melt['Medal'] == 'Gold']['Count'],
        name='Gold',
        marker_color='#FFD700',
        text=tmpS_melt[tmpS_melt['Medal'] == 'Gold']['Count'].astype(str) + ' Gold Medals',
        hoverinfo='text'
    ))
    
    # Add a trace for Silver Medals
    fig.add_trace(go.Bar(
        x=tmpS_melt[tmpS_melt['Medal'] == 'Silver']['NOC'],
        y=tmpS_melt[tmpS_melt['Medal'] == 'Silver']['Count'],
        name='Silver',
        marker_color='#C0C0C0',
        text=tmpS_melt[tmpS_melt['Medal'] == 'Silver']['Count'].astype(str) + ' Silver Medals',
        hoverinfo='text'
    ))
    
    # Add a trace for Bronze Medals
    fig.add_trace(go.Bar(
        x=tmpS_melt[tmpS_melt['Medal'] == 'Bronze']['NOC'],
        y=tmpS_melt[tmpS_melt['Medal'] == 'Bronze']['Count'],
        name='Bronze',
        marker_color='#CD7F32',
        text=tmpS_melt[tmpS_melt['Medal'] == 'Bronze']['Count'].astype(str) + ' Bronze Medals',
        hoverinfo='text'
    ))
    
    # Add a trace for the total of Gold, Silver and Bronze Medals
    tmpS_melt['Total'] = tmpS_melt.groupby('NOC')['Count'].transform('sum')
    fig.add_trace(go.Bar(
        x=tmpS_melt['NOC'].unique(),
        y=tmpS_melt[tmpS_melt['Medal'] == 'Gold']['Total'],
        name='Total Medals',
        marker_color='pink',
        text=tmpS_melt[tmpS_melt['Medal'] == 'Gold']['Total'].astype(str) + ' Total Medals',
        hoverinfo='text'
    ))
    
    # set the y-axis label
    fig.update_layout(yaxis_title='Total Medals')
    
    # set the title of the chart
    fig.update_layout(title='Total Medals by Region')
    
    # display the chart using streamlit.plotly_chart
    st.plotly_chart(fig)


# Define the function to create the bar chart
def create_bar_chartT(dataframe):
    # Sort the dataframe by the 'Gold' column in descending order
    dataframe = dataframe.sort_values(['Gold','Silver','Bronze'], ascending=True)
    # Create a horizontal stacked bar chart using Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dataframe['Gold'], y=dataframe['region'], name='Gold', orientation='h',marker=dict(color='gold')))
    fig.add_trace(go.Bar(x=dataframe['Silver'], y=dataframe['region'], name='Silver', orientation='h',marker=dict(color='silver')))
    fig.add_trace(go.Bar(x=dataframe['Bronze'], y=dataframe['region'], name='Bronze', orientation='h',marker=dict(color='brown')))
    
    # Set chart title and axis labels, and adjust chart height
    fig.update_layout(
        title='Medal Tally by Region',
        xaxis=dict(title='Medals'),
        yaxis=dict(title='Region'),
        barmode='stack',  # stack bars
        height=5000  # set chart height to 4000 pixels
    )
    
    # Display the chart using Streamlit
    st.plotly_chart(fig)
# Define the function to create the line chart
def create_line_chartT(dataframe):
    # Create a line chart using Plotly
    fig = go.Figure()
    
    # Add traces for each medal type
    fig.add_trace(go.Scatter(x=dataframe['Year'], y=dataframe['Gold'], name='Gold', line=dict(color='gold'), mode='lines+markers'))
    fig.add_trace(go.Scatter(x=dataframe['Year'], y=dataframe['Silver'], name='Silver', line=dict(color='silver'), mode='lines+markers'))
    fig.add_trace(go.Scatter(x=dataframe['Year'], y=dataframe['Bronze'], name='Bronze', line=dict(color='brown'), mode='lines+markers'))
    fig.add_trace(go.Scatter(x=dataframe['Year'], y=dataframe['total'], name='Total', line=dict(color='black'), mode='lines+markers'))
    
    # Set chart title and axis labels
    fig.update_layout(
        title='Medal Tally by Year',
        xaxis=dict(title='Year'),
        yaxis=dict(title='Total Medals'),
        height=500  # set chart height to 500 pixels
    )
    
    # Display the chart using Streamlit's Plotly chart component
    st.plotly_chart(fig)


def jpn21():
    # Set background color
    background_color = "#f0f0f0"
    
    # Create plot using Matplotlib
    fig, ax = plt.subplots(figsize=(4, 5), facecolor=background_color)
    
    temp = df_21_full[:15].sort_values(by='Total')
    my_range=range(1,len(df_21_full[:15]['Team/NOC'])+1)
    
    ax.set_facecolor(background_color)
    
    plt.hlines(y=my_range, xmin=0, xmax=temp['Total'], color='gray')
    plt.plot(temp['Total'], my_range, "o", markersize=10, color='#244747')
    plt.plot(temp['Total'][2], my_range[10], "o", markersize=20, color='#B73832')
    
    Xstart, Xend = ax.get_xlim()
    Ystart, Yend = ax.get_ylim()
    
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.set_xlabel("Total Medals", fontfamily='monospace', loc='left', color='gray')
    ax.set_axisbelow(True)
    
    for s in ['top', 'right', 'bottom', 'left']:
        ax.spines[s].set_visible(False)
    
    ax.text(-90, Yend+2.3, 'Olympic Total Medals by Country: Tokyo 2021', fontsize=15, fontweight='bold', fontfamily='serif', color='#323232')
    ax.text(-90, Yend+1.1, 'Japan hosted the games for the second time', fontsize=10, fontweight='bold', fontfamily='sansserif', color='#B73832')
    
    # Add titles and axis names
    plt.yticks(my_range, temp['Team/NOC'])
    plt.xlabel('')
    
    ax.annotate(temp['Total'][2], xy=(54.86,10.95), va='center', ha='left', fontweight='light', fontfamily='monospace', fontsize=10, color='white', rotation=0)
    
    # Display the plot using Streamlit
    st.pyplot(fig)
    
def hst():
    # Set background color
    background_color = '#F5F5F5'
    
    # Create figure and axis objects
    fig, ax = plt.subplots(1,1, figsize=(10,4), facecolor=background_color)
    
    # Create scatterplot
    sns.scatterplot(data=df_new.query("Is_Host == 0"), x='Year', y='Total Medals', s=45, ec='black', color='#244747',ax=ax)
    sns.scatterplot(data=df_new.query("Is_Host == 1"), x='Year', y='Total Medals', s=75, ec='black', color='#B73832',ax=ax)
    
    # Get x and y limits
    Xstart, Xend = ax.get_xlim()
    Ystart, Yend = ax.get_ylim()
    
    # Set tick parameters, labels, and facecolor
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.set_ylabel("Total Medals",fontfamily='monospace',loc='bottom',color='gray')
    ax.set_xlabel("")
    ax.set_facecolor(background_color)
    ax.set_axisbelow(True)
    
    # Hide spines
    for s in ['top','right','bottom','left']:
        ax.spines[s].set_visible(False)
        
    # Add text annotations
    ax.text(Xstart,Yend+80, 'Olympic Medals by Country: Hosting always helps', fontsize=15,fontweight='bold',fontfamily='serif',color='#323232')
    ax.text(Xstart,Yend+40, 'Host Medals', fontsize=10,fontweight='bold',fontfamily='sansserif',color='#B73832')
    ax.text(Xstart,Yend+5, 'Others', fontsize=10,fontweight='bold',fontfamily='sansserif',color='#244747')
    
    # Show plot
    st.pyplot(fig)


def olympic_medals_by_country(df_new):
    background_color = "#F4F4F4"
    fig, ax = plt.subplots(1,1, figsize=(11, 5), facecolor=background_color)

    # top 20
    top_list_ = df_new.groupby('region')['Total Medals'].mean().sort_values(ascending=False).reset_index()[:20].sort_values(by='Total Medals',ascending=True)

    plot = 1
    for country in top_list_['region']:
        mean = df_new[df_new['region'] == country].groupby('region')['Total Medals'].mean()
        # historic scores
        sns.scatterplot(data=df_new[df_new['region'] == country], y=plot, x='Total Medals',color='lightgray',s=50,ax=ax)
        # mean score
        sns.scatterplot(data=df_new[df_new['region'] == country], y=plot, x=mean,color='#244747',ec='black',linewidth=1,s=75,ax=ax)
        # Hosting score
        sns.scatterplot(data=(df_new[(df_new['region'] == country) & (df_new['Is_Host'] == 1)]), y=plot, x='Total Medals',color='#B73832',ec='black',linewidth=1,s=75,ax=ax)   
        plot += 1

    Xstart, Xend = ax.get_xlim()
    Ystart, Yend = ax.get_ylim()

    ax.set_yticks(top_list_.index+1)
    ax.set_yticklabels(top_list_['region'][::-1], fontdict={'horizontalalignment': 'right'}, alpha=0.7)
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.set_xlabel("Total Medals",fontfamily='monospace',loc='left',color='gray')
    ax.set_facecolor(background_color)
    ax.hlines(y=top_list_.index+1, xmin=0, xmax=Xend, color='gray', alpha=0.5, linewidth=.3, linestyles='--')
    ax.set_axisbelow(True)

    for s in ['top','right','bottom','left']:
        ax.spines[s].set_visible(False)

    ax.text(0,Yend+3.5, 'Olympic Medals by Country: Hosts through the years', fontsize=15,fontweight='bold',fontfamily='serif',color='#323232')
    ax.text(0,Yend+2.1, 'Hosting', fontsize=10,fontweight='bold',fontfamily='sansserif',color='#B73832')
    ax.text(0,Yend+1, 'Average', fontsize=10,fontweight='bold',fontfamily='sansserif',color='#244747')

    st.pyplot(fig)


def overallcomphost(df_new):
    host_list = list(df_new.query("Is_Host == 1")['Host_Country'].value_counts().index)

    Not_hosting = df_new[df_new['region'].isin(host_list)].query("Is_Host == 0")[['Bronze','Silver','Gold']].mean().reset_index()
    hosting = df_new[df_new['region'].isin(host_list)].query("Is_Host == 1")[['Bronze','Silver','Gold']].mean().reset_index()
    
    radar = pd.merge(hosting, Not_hosting, on='index')
    radar.columns = ['Medal','Hosting', 'Not Hosting']
    radar = radar.set_index('Medal').T.reset_index()
    radar = radar[['index','Gold','Silver','Bronze']]

    # Set the background color
    background_color = "#f2efe8"
    
    # Sample data
    radar = pd.DataFrame({
        'index': ['Host', 'Not Host'],
        'Gold': [23, 17],
        'Silver': [20, 21],
        'Bronze': [27, 19]
    })
    
    # Convert the index to a list
    Comparison = radar["index"].values.tolist()
    
    # List of medal types
    medals = ['Gold', 'Silver', 'Bronze']
    
    # Length of medal types
    length = len(medals)
    
    # Colors for the radar chart
    colors = ["#B73832", "#244747"]
    
    # The angles at which the values of the numeric variables are placed
    ANGLES = [n / length * 2 * np.pi for n in range(length)]
    ANGLES += ANGLES[:1]
    
    # Angle values going from 0 to 2*pi
    HANGLES = np.linspace(0, 2 * np.pi)
    
    # Surrounding circles
    H0 = np.ones(len(HANGLES)) * 20
    H1 = np.ones(len(HANGLES)) * 40
    H2 = np.ones(len(HANGLES)) * 60
    
    # Rotate the plot
    theta_offset = np.pi / 2.6
    theta_direction = -1
    
    # Create the plot
    fig = plt.figure(figsize=(6, 6), facecolor=background_color)
    ax = fig.add_subplot(1,1,1, polar=True)
    ax.set_facecolor(background_color)
    
    # Set the radius of the plot
    ax.set_ylim(-0.1, radar[['Gold', 'Silver', 'Bronze']].max().max() + 5)
    
    # Plot the radar chart
    for idx, host in enumerate(Comparison):
        values = radar.iloc[idx].drop("index").values.tolist()
        values += values[:1]
        ax.plot(ANGLES, values, c=colors[idx], linewidth=1.5, label=host)
        ax.scatter(ANGLES, values, s=160, c=colors[idx], ec='white', zorder=10)
    
    # Edit lines & fill between
    ax.plot(HANGLES, H0, ls=(0, (3, 1)), lw=1, c='#b3b3b3')
    ax.plot(HANGLES, H1, ls=(0, (5, 3)), lw=1, c='#b3b3b3')
    ax.plot(HANGLES, H2, ls=(0, (6, 4)), lw=1, c='#b3b3b3')
    ax.fill(HANGLES, H2, '#f2efe8')
    
    # Tidy up & labels
    ax.set_xticks(ANGLES[:-1])
    ax.set_xticklabels(medals, size=10, color='#323232')
    ax.set_yticks([])
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.spines["start"].set_color("none")
    ax.spines["polar"].set_color("none")
    
    # Set the title and subtitle
    fig.text(0, 0.98, 'Average Olympic Medals:\nIs this difference significant?', fontsize=15, fontweight='bold', fontfamily='serif', color='#323232')
    fig.text(0, 0.93, 'Hosting', fontsize=10, fontweight='bold', fontfamily='sansserif', color='#B73832')
    fig.text(0, 0.89, 'Not Hosting', fontsize=10, fontweight='bold', fontfamily='sansserif',color='#B73832')
    fig.text(0,0.89, 'Not Hosting', fontsize=10,fontweight='bold',fontfamily='sansserif',color='#244747')
    st.pyplot(fig)

def malefemale1(dfMF,sport_choice):
    # df1=dfMF.loc[dfMF['Sport']==sport_choice]
    # Group the merged dataframe by year and sex, and count the number of unique athletes for each year and sex
    df_grouped = dfMF.groupby(["Year", "Sex"])["ID"].nunique().reset_index()
    
    # Separate the male and female data
    df_male = df_grouped[df_grouped["Sex"] == "M"]
    df_female = df_grouped[df_grouped["Sex"] == "F"]
    
    # Define the size of the bubbles
    male_sizes = np.sqrt(df_male["ID"] * 2)
    female_sizes = np.sqrt(df_female["ID"] * 2)
    
    # Plot data
    fig, ax = plt.subplots()
    
    ax.scatter(df_male["Year"], df_male["ID"], label="Male", alpha=0.8, s=male_sizes, color='darkblue')
    ax.scatter(df_female["Year"], df_female["ID"], label="Female", alpha=0.8, s=female_sizes, color='red')
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Participants")
    ax.set_title("Number of Participants by Year")
    ax.legend()
    ax.grid(False)
    
    # Function to display values on hover
    def on_plot_hover(event):
        for i in range(len(df_grouped)):
            if ax.collections[i].contains(event)[0]:
                year = df_grouped.iloc[i]["Year"]
                sex = df_grouped.iloc[i]["Sex"]
                num_participants = df_grouped.iloc[i]["ID"]
                ax.annotate(f"{year}: {num_participants} {sex}", xy=(df_grouped.iloc[i]["Year"], df_grouped.iloc[i]["ID"]), 
                            xytext=(df_grouped.iloc[i]["Year"]-1, df_grouped.iloc[i]["ID"]+300), 
                            bbox=dict(facecolor='white', alpha=0.8))
    
    fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)
    
    # Display plot in Streamlit app
    st.pyplot(fig)


# Create a function that plots the number of medals by age for a specific sport
def plot_medals_by_age(sport,dfMF):
    # Filter the data to include only the specified sport and gold, silver, and bronze medals
    df_sport = dfMF[dfMF['Sport'] == sport]
    df_male = df_sport[df_sport['Sex'] == 'M']
    df_female = df_sport[df_sport['Sex'] == 'F']

    df_male_medal_counts = df_male.groupby('Age')['Medal'].count()
    df_female_medal_counts = df_female.groupby('Age')['Medal'].count()

    # Create a line graph with 3 lines for gold, silver, and bronze medals
    fig, ax = plt.subplots()
    ax.plot(df_male_medal_counts.index, df_male_medal_counts.values, label='Male',alpha=1.0,color='red',linestyle='--')
    ax.plot(df_female_medal_counts.index, df_female_medal_counts.values, label='Female',alpha=1.0,color='darkblue',linestyle='--')
    
    ax.scatter(df_male_medal_counts.index, df_male_medal_counts.values,alpha=1.0,color='red',linestyle='--')
    ax.scatter(df_female_medal_counts.index, df_female_medal_counts.values,alpha=1.0,color='darkblue',linestyle='--')

    # Set the plot title and axis labels
    ax.set_title(f'Number of Medals by Age for {sport}')
    ax.set_xlabel('Age')
    ax.set_ylabel('Number of Medals')

    # Add a legend to the plot
    ax.legend()
    
    ax.grid(False)

    # Show the plot using st.pyplot()
    st.pyplot(fig)

#@st.cache
def calculate_weight_height_proportions(dfMF, sex):
    # Filter the data to include only the specified sex
    df_filtered = dfMF[dfMF['Sex'] == sex]
    
    # Calculate the weight-to-height proportion for each athlete
    df_filtered['proportions'] = df_filtered['Weight'] / (df_filtered['Height'] / 100) ** 2

    # Group the data by year and calculate the median proportion for each year
    df_median = df_filtered.groupby('Year')['proportions'].median()

    return df_median

def create_bmi_plot(sport,dfMF):

    # Calculate the weight-to-height proportions for male and female athletes
    df_filter = dfMF[dfMF['Sport'] == sport]
    
    proportions_male = calculate_weight_height_proportions(df_filter, 'M')
    proportions_female = calculate_weight_height_proportions(df_filter, 'F')

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the median proportions for male and female athletes
    ax.scatter(proportions_male.index, proportions_male.values, label='Male',alpha=1.0, color='red')
    ax.scatter(proportions_female.index, proportions_female.values, label='Female',alpha=1.0, color='darkblue')

    ax.plot(proportions_male.index, proportions_male.values,alpha=1.0, color='red')
    ax.plot(proportions_female.index, proportions_female.values,alpha=1.0, color='darkblue')

    # Set the plot title and axis labels
    ax.set_title(f'BMI (Male vs Female) for {sport}')
    ax.set_xlabel('Year')
    ax.set_ylabel('BMI')

    # Set the y-axis limits to match the range of proportions
    proportions = pd.concat([proportions_male, proportions_female])
    ax.set_ylim(proportions.min() * 0.9, proportions.max() * 1.1)

    # Add a legend to the plot
    ax.legend()

    # Show the plot
    st.pyplot(fig)

# Create a function to generate the bar chart
def create_medal_count_plot(sport,dfMF):
    dfMF['region'] = dfMF['region'].fillna('Unknown')
        # filter the DataFrame by sport
    df_sport = dfMF[dfMF['Sport'] == sport]
    
    # group the DataFrame by country and medal type, and aggregate the count of medals
    df_grouped = df_sport.groupby(['region', 'Medal']).size().unstack(fill_value=0)
    
    # create a new column with the total medal count for each country
    df_grouped['Total'] = df_grouped.sum(axis=1)
    
    # sort the DataFrame by total medal count
    df_sorted = df_grouped.sort_values('Total', ascending=False)
    
    df_sorted = df_sorted[df_sorted['Total'] > 10] 

    # create a list of the medal types in the order gold, silver, bronze
    medal_types = ['Gold', 'Silver', 'Bronze']
    
    # create a list of colors for the bars, corresponding to the medal types
    colors = ['#FFD700', '#C0C0C0', '#CD7F32']
#     sns.color_palette("Set2")
    # create a bar chart showing the medal counts for each country
    fig, ax = plt.subplots(figsize=(12, 10))
    for i, medal_type in enumerate(medal_types):
        ax.barh(df_sorted.index, df_sorted[medal_type], label=medal_type, color=colors[i])
    
    # set plot title and axis labels
    plt.title(f'Medal Count by Country - {sport}')
    plt.xlabel('Number of Medals')
    plt.ylabel('Country')
    
    # add legend
    ax.legend(loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    # adjust layout to make room for legend
    plt.subplots_adjust(right=0.8)
    
    # show plot
    st.pyplot(fig)



def create_medal_heatMap_plot(sport,dfMF):
    dfMF['region'] = dfMF['region'].fillna('Unknown')
    # Filter the data for the selected sport
    if isinstance(sport, list):
      
        df_sports = dfMF[dfMF['Sport'].isin(sport)]
    else: 
        df_sports = dfMF[dfMF['Sport'] == sport]
    
    df_sports['Medal'].fillna(0, inplace=True)

    # Pivot the data to create a heatmap
    df_heatmap = df_sports.pivot_table(index='Age', columns=['Sport', 'Sex'], values='Medal', aggfunc='count')

    # Create the heatmap
    fig = px.imshow(df_heatmap,
                    labels=dict(x='Sport', y='Age', color='Medal Count'),
                    x=df_heatmap.columns.get_level_values(1),
                    y=df_heatmap.index,
                    color_continuous_scale='reds',
                    title=f'Medal Count by Age and Gender for {sport} ')
    # Show the plot
    st.plotly_chart(fig)

def create_3d_scatter_plot(sport,dfMF):
    if isinstance(sport, list):
      
        df_filtered = dfMF[dfMF['Sport'].isin(sport)]
    else: 
        df_filtered = dfMF[dfMF['Sport'] == sport]
    
    # df_filtered = dfMF[dfMF['Sport'] == sport]
    df_grouped = df_filtered.groupby('Sport').agg({'Age': 'mean', 'Weight': 'mean', 'Height': 'mean'})
    fig = px.scatter_3d(df_grouped, x='Age', y='Height', z='Weight', color=df_filtered['Sport'].unique())
    st.plotly_chart(fig)

def create_bubble_plot(sport,dfMF):
    # Create separate columns for Gold, Silver, and Bronze medals
    df_medals = pd.get_dummies(dfMF['Medal'])

    # Add up the three columns to get total medals won
    df_filtered = dfMF[ dfMF['Sport'] == sport ]
    
    df_filtered['Total Medals'] = df_medals.sum(axis=1)
    
    df_country = df_filtered.groupby(['NOC', 'Year', 'Sex']).agg({'Height': 'mean', 'Weight': 'mean', 'Total Medals': 'sum'}).reset_index()

    fig = px.scatter(df_country, x='Weight', y='Height', size='Total Medals', color='Sex',
                     hover_name='NOC',title='Weight vs Height vs Total Medals by Country and Gender')
    
    fig.update_layout(xaxis_title='Weight', yaxis_title='Height', legend_title='Sex')
    st.plotly_chart(fig)

def top_countries_by_sex(data):
    topn = 10
    male = data[data.Sex=='M']
    female = data[data.Sex=='F']
    count_male = male.dropna().NOC.value_counts()[:topn].reset_index()
    count_female = female.dropna().NOC.value_counts()[:topn].reset_index()

    pie_men = go.Pie(labels=count_male['index'],values=count_male.NOC,name="Men",hole=0.4,domain={'x': [0,0.46]})
    pie_women = go.Pie(labels=count_female['index'],values=count_female.NOC,name="Women",hole=0.4,domain={'x': [0.5,1]})

    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    fig.add_trace(pie_men, 1, 1)
    fig.add_trace(pie_women, 1, 2)

    fig.update_layout(title = 'Top-10 countries with medals by sex', font=dict(size=15), legend=dict(orientation="h"),
                      annotations = [dict(x=0.2, y=0.5, text='Men', showarrow=False, font=dict(size=20)),
                                     dict(x=0.8, y=0.5, text='Women', showarrow=False, font=dict(size=20)) ])
    
    st.plotly_chart(fig)

def display_scatterplot(data):
    tmp = data.groupby(['Sport'])['Height', 'Weight'].agg('mean').dropna()
    df1 = pd.DataFrame(tmp).reset_index()
    tmp = data.groupby(['Sport'])['ID'].count()
    df2 = pd.DataFrame(tmp).reset_index()
    dataset = df1.merge(df2) #DataFrame with columns 'Sport', 'Height', 'Weight', 'ID'

    scatterplots = list()
    for sport in dataset['Sport']:
        df = dataset[dataset['Sport']==sport]
        trace = go.Scatter(
            x = df['Height'],
            y = df['Weight'],
            name = sport,
            marker=dict(
                symbol='circle',
                sizemode='area',
                sizeref=10,
                size=df['ID'])
        )
        scatterplots.append(trace)

    layout = go.Layout(title='Mean height and weight by sport', 
                       xaxis=dict(title='Height, cm'), 
                       yaxis=dict(title='Weight, kg'),
                       showlegend=True)

    fig = dict(data = scatterplots, layout = layout)
    st.plotly_chart(fig)

def create_stacked_bar_country_plot(noc, year, df):
    df_noc_year = df[(df['NOC'] == noc) & (df['Year'] == year)]
    
    # Group the data by Sport, Sex, and Medal
    df_grouped = df_noc_year.groupby(['Year', 'Sport', 'Sex', 'Medal'], as_index=False)['ID'].count()

    # Pivot the data to create the stacked bar chart
    df_pivot = df_grouped.pivot(index=['Year', 'Sport'], columns='Sex', values=['Gold', 'Silver', 'Bronze'])
    df_pivot.reset_index(inplace=True)
    df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]

    if df_pivot.empty:
        st.warning(f"No medals were won by {noc} in {year}")
    else:
        # Create the stacked bar chart
        fig = px.bar(df_pivot, x='Sport_', y=['Male_Gold', 'Male_Silver', 'Male_Bronze', 'Female_Gold', 'Female_Silver', 'Female_Bronze'], barmode='stack',
                     animation_frame='Year_', hover_name='Sport_')

        # Update the layout
        fig.update_layout(title=f'Total Medals Won by Male and Female Athletes for {noc} in {year}',
                          xaxis_title='Sport',
                          yaxis_title='Total Medals')

        # Show the plot
        st.plotly_chart(fig)



st.sidebar.title("Olympics Data Analysis")
st.sidebar.image('https://i.pinimg.com/originals/27/07/eb/2707ebe3f9114547b13a6ad01daf5f51.png')

user_menu = st.sidebar.radio(
    'Select an Option',
    ('Home','Medal Tally Overall','Overall General Analysis','Country-wise Analysis','Athlete wise Analysis','Male v Female Analysis','Host-country Advantage')
)

if user_menu == 'Home':
    st.title("Historic Olympic Data Analysis")
    st.image('https://akm-img-a-in.tosshub.com/indiatoday/images/story/202108/AP21219462944826_1200x768.jpeg?size=690:388')
    st.title("Medal Distribution Overall:")
    colors = ['#f4cb42', '#cd7f32', '#a1a8b5'] #gold,bronze,silver
    medal_counts = dftmp.Medal.value_counts(sort=True)
    labels = medal_counts.index
    values = medal_counts.values

    pie = go.Pie(labels=labels, values=values, marker=dict(colors=colors))
    layout = go.Layout(title='Medal distribution')
    fig = go.Figure(data=[pie], layout=layout)

    st.plotly_chart(fig)

if user_menu == 'Medal Tally Overall':
    st.sidebar.header("Medal Tally")
    years,country = helper.country_year_list(df)

    selected_year = st.sidebar.selectbox("Select Year",years)
    selected_country = st.sidebar.selectbox("Select Country", country)
    medal_tally = helper.fetch_medal_tally(df,selected_year,selected_country)
    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title("Overall Tally")
        # Create a bar chart to visualize the medal tally data
        create_bar_chartT(medal_tally)
    if selected_year != 'Overall' and selected_country == 'Overall':
        st.title("Medal Tally in " + str(selected_year) + " Olympics")
        # Create a bar chart to visualize the medal tally data
        create_bar_chartT(medal_tally)
    if selected_year == 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " overall performance")
        # Create a line chart to visualize the medal tally data
        create_line_chartT(medal_tally)
    if selected_year != 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " performance in " + str(selected_year) + " Olympics")
    
    

    st.title("Top countries total medals with time:")
    #st.table(medal_tally)
    fig = px.bar(df_melted, x='value', y='variable', color='variable', animation_frame='Year', orientation='h',height=600)
    # Update the x and y axis labels
    fig.update_xaxes(title_text='Total Medals')
    fig.update_yaxes(title_text='NOC regions')
    # Render the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    st.title("Geographic distribution of medals overall(1896-2016):")
    wmap0()

if user_menu == 'Overall General Analysis':
    editions = df['Year'].unique().shape[0] - 1
    cities = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events = df['Event'].unique().shape[0]
    athletes = df['Name'].unique().shape[0]
    nations = df['region'].unique().shape[0]

    st.title("Top Statistics")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.header("Editions")
        st.title(editions)
    with col2:
        st.header("Hosts")
        st.title(cities)
    with col3:
        st.header("Sports")
        st.title(sports)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Events")
        st.title(events)
    with col2:
        st.header("Nations")
        st.title(nations)
    with col3:
        st.header("Athletes")
        st.title(athletes)

    st.title("Geographic distribution of medals overall Gender and Sports wise:")
    wmap()

    st.title("Top countries medals wise:")
    wmap2()

    nations_over_time = helper.data_over_time(df,'region')
    fig = px.line(nations_over_time, x="Edition", y="region",markers=True)
    fig.update_layout(yaxis_title="Number of countries")
    st.title("Participating Nations over the years")
    st.plotly_chart(fig)

    events_over_time = helper.data_over_time(df, 'Event')
    fig = px.line(events_over_time, x="Edition", y="Event", markers=True)
    st.title("Events over the years")
    st.plotly_chart(fig)

    athlete_over_time = helper.data_over_time(df, 'Name')
    fig = px.line(athlete_over_time, x="Edition", y="Name", markers=True)
    fig.update_layout(yaxis_title="Number of Athletes")
    st.title("Athletes over the years")
    st.plotly_chart(fig)

    st.title("No. of Events over time(Every Sport)")
    fig,ax = plt.subplots(figsize=(20,20))
    x = df.drop_duplicates(['Year', 'Sport', 'Event'])
    ax = sns.heatmap(x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype('int'),
                annot=True)
    st.pyplot(fig)

    # Create a dropdown menu to select a specific sport
    sports_list = dfMF['Sport'].unique()
    selected_sport = st.selectbox('Select a sport', sports_list)

    # Call the function to create the bar chart
    create_medal_count_plot(selected_sport,dfMF)

    st.title("Most successful Athletes")
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0,'Overall')

    selected_sport = st.selectbox('Select a Sport',sport_list)
    x = helper.most_successful(df,selected_sport)
    st.table(x)

if user_menu == 'Country-wise Analysis':

    st.sidebar.title('Country-wise Analysis')

    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()

    selected_country = st.selectbox('Select a Country',country_list)

    country_df = helper.yearwise_medal_tally(df,selected_country)
    fig = px.line(country_df, x="Year", y="Medal",markers=True)
    st.title(selected_country + " Medal Tally over the years")
    st.plotly_chart(fig)

    st.title(selected_country + " excels in the following sports")
    pt = helper.country_event_heatmap(df,selected_country)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax = sns.heatmap(pt,annot=True)
    st.pyplot(fig)

    st.title("Top 10 athletes of " + selected_country)
    top10_df = helper.most_successful_countrywise(df,selected_country)
    st.table(top10_df)

if user_menu == 'Athlete wise Analysis':
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])
    famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']
    select_sports = st.multiselect('Select the sports',
                              famous_sports,
                              ['Athletics','Swimming','Tennis','Golf'])
    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[(athlete_df['Medal'] == 'Gold') & (athlete_df['Sport'].isin(select_sports))]['Age'].dropna()
    x3 = athlete_df[(athlete_df['Medal'] == 'Silver') & (athlete_df['Sport'].isin(select_sports))]['Age'].dropna()
    x4 = athlete_df[(athlete_df['Medal'] == 'Bronze') & (athlete_df['Sport'].isin(select_sports))]['Age'].dropna()

    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],show_hist=False, show_rug=False)
    fig.update_layout(autosize=False,width=1000,height=600)
    st.title("Distribution of Age")
    st.plotly_chart(fig)

    x = []
    name = []
    
    for sport in select_sports:
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
        name.append(sport)

    fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=1000, height=600)
    st.title("Distribution of Age wrt Sports(Gold Medalist)")
    st.plotly_chart(fig)

    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    st.title("Sportswise medal winning comparison with age:")
    # Create a dropdown menu to select a specific sport
    sports_list = dfMF['Sport'].unique()
    # selected_sport = st.selectbox('Select a Sport', sports_list)

    # Call the create_medal_heatMap_plot function with the selected sport
    create_medal_heatMap_plot(select_sports,dfMF)
    
    st.title("Sportswise medal winning comparison with age, weight and height:")
    # Create a dropdown menu to select a specific sport
    sports_list1 = dfMF['Sport'].unique()
    # selected_sport1 = st.selectbox('Select a Sport below', sports_list1)

    # Call the create_3d_scatter_plot function with the selected sport
    create_3d_scatter_plot(select_sports,dfMF)

    st.title("Sportswise medal winnings overall based on weight and height:")
    display_scatterplot(dftmp)
if user_menu == 'Male v Female Analysis':
    country,year,age,bmi,hw = st.tabs(
    ["Country Comparison",'Yearly Participation', 'Age Comparison', 'BMI Comparison','Height & Weight Comparison'])
    sports_list = dfMF['Sport'].unique()

    with country:
        st.title("Top 10 countries M v F")
        top_countries_by_sex(dftmp)
    with year:
        # sport_choice1 = st.selectbox('Select a sport for yearly comparison', sports_list)
        # st.title("Participation with time Male v Female")
        st.title("Participation with time Male v Female")
        malefemale1(dfMF,'na')
    with age:
        st.title("Age comparison sportswise Male v Female")
        st.subheader("Please select a sport from sidebar")
        # Create a dropdown menu to select a specific sport
        sports_list = dfMF['Sport'].unique()
        sport_choice = st.selectbox('Select a sport for age:', sports_list)

    # Display plot based on selected sport
        plot_medals_by_age(sport_choice,dfMF)
    with bmi:
        st.title("Men Vs Women BMI Over the Years")
        # Get the list of sports
        sports_list = dfMF['Sport'].unique()

        # Create a dropdown menu to select a specific sport
        selected_sport = st.selectbox('Select a sport for BMI', sports_list)

        # Create the plot for the selected sport
        create_bmi_plot(selected_sport,dfMF)
    with hw:
        st.title("Sportswise medal winnings countrywise based on weight and height:")
        # Create a dropdown menu to select a specific sport
        sports_list2 = dfMF['Sport'].unique()
        selected_sport2 = st.selectbox('Select any sport', sports_list2)
        create_bubble_plot(selected_sport2,dfMF)
    





if user_menu == 'Host-country Advantage':
    st.title("Host-country Advantage")
    st.text("In the recent Olympics, Japan did extremely well while hosting the event.\nAs they were the host nation of this Olympics, an obvious question was:\nDoes hosting the Olympics improve performance in the medals table?\nTo answer this, we'll perform an exploratory data analysis \nalong with some statistical tests and visualizations.\nWe'll use past Olympic data(1896-2016), as well as \ndata from the Tokyo Olympics(2021).")
    jpn21()
    st.title("Host-country v Non Host country performance")
    hst()
    # create the Streamlit app
    st.title("Olympic Medals by Country: Hosts through the years")
    olympic_medals_by_country(df_new)
    st.title("Olympic Medals Overall Performance Host v Non Host")
    overallcomphost(df_new)
