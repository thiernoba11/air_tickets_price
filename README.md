# Airline ticket price dataset

There are two airline ticket price datasets. Dataset1 contains 8 routes, dataset2 contains 12 routes as shown following:

```
dataset1 = ["BCN_BUD",  # route 1  Barcelona -> Budapest
          "BUD_BCN",  # route 2    Budapest -> Barcelona
          "CRL_OTP",  # route 3    Brussels -> Bucharest
          "MLH_SKP",  # route 4    Mulhouse -> Skopje
          "MMX_SKP",  # route 5    Sweden -> Skopje
          "OTP_CRL",  # route 6    Bucharest -> Brussels
          "SKP_MLH",  # route 7    Skopje -> Mulhouse
          "SKP_MMX"]  # route 8    Skopje -> Sweden
```

```
dataset2 = ["BGY_OTP", # route 1  Milan -> Bucharest
            "BUD_VKO", # route 2  Budapest -> Moscow
            "CRL_OTP", # route 3  Brussels -> Bucharest
            "CRL_WAW", # route 4  Brussels -> Warsaw
            "LTN_OTP", # route 5  London -> Bucharest
            "LTN_PRG", # route 6  London -> Prague
            "OTP_BGY", # route 7  Bucharest -> Milan
            "OTP_CRL", # route 8  Bucharest -> Brussels
            "OTP_LTN", # route 9  Bucharest -> London
            "PRG_LTN", # route 10 Prague -> London
            "VKO_BUD", # route 11 Moscow -> Budapest
            "WAW_CRL"] # route 12 Warsaw -> Brussels
```

All the files contained in the datasets are in json format. And the data are observed from November 09, 2015 to May 09, 2016. 

## Sample data entry:

```
the query data is the folder name
|-"ArrivalStationCode":"BUD"     # code for the arrival station 
|-"CurrentDate":"30\/12\/2015"   # departure date
|-"Date":"20151230"              # departure date
|-"DepartureStationCode":"BCN"   # code for departure station
|-"Flights":[{
    |-"CarrierCode":"W6"         # NA
    |-"FlightNumber":"2376"      # flight number
    |-"STD":"09:15"              # departure time
    |-"STA":"11:50"              # arrival time
    |-"ArrivalStationName":"Budapest"             # the city name for the arrival station
    |-"DepartureStationName":"Barcelona El Prat"  # the city name for the departure station
    |-"IsMACStation":"True"                       # NA
    |-"IsAirportChange":"False"}]                 # indicator whether need to change (all false in the dataset)
|-"HasSelection":"True"
|-"InMonth":"True"
|-"MinimumPrice":"€49.99"                         # price

```

## Sample feature extraction code in python
```
|-sample_data_extraction.py
```

### Extracted feature 1
```
feature 1: flight number -> dummy variables
feature 2: departure date interval from "20151109", because the first observed date is 20151109
feature 3: observed days before departure date
feature 4: minimum price before the observed date
feature 5: maximum price before the observed date
output: the date which has the minimum price from the observed date to departure date is set to 1;
        other entries set to 0
```

### Extracted feature 2
```
feature 1: flight number -> dummy variables
feature 2: departure date interval from "20151109", because the first observed date is 20151109
feature 3: observed days before departure date
feature 4: minimum price before the observed date
feature 5: maximum price before the observed date
output: the ticket price for each entry
```

## Announcement
This dataset is only used for research purpose, you cannot publish for commercial usage. All rights reserved.

## Citation
```
@Misc{jun2016airdata,
  author =   {Jun Lu},
  title =    {Dataset for airline ticket price},
  howpublished = {\url{https://github.com/lujunzju/data_airticket}},
  year = {since 2016}
}
```


## Lincense
MIT
