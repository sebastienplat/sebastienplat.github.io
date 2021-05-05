[Kaggle Competition link](https://www.kaggle.com/c/santander-product-recommendation)

# EXPLORATORY DATA ANALYSIS

#### Goal

> Santander is challenging Kagglers to predict which products their existing customers will use in the next month, based on their past behavior and that of similar customers.

> With a more effective recommendation system in place, Santander can better meet the individual needs of all customers and ensure their satisfaction no matter where they are in life.


First thing is to understand the problem.

#### First Look

Train set:

+ Extremely large dataset: 13.6 million rows
+ 950k unique customers
+ 17 months history, from 2015-01 to 2016-05

<div class='row'>
<div class='col-md-6'>
<div google-chart chart='{
options: {
  title: "Number of months history per customer",
  legend: {position: "none"},
  vAxis: { viewWindow: { min: 0, max: 1000000} }
},
type: "ColumnChart",
data: {
					"cols": [
						{id: "t", label: "Seniority", type: "string"},
						{id: "s", label: "Rows", type: "number"}
					], 
					"rows": [
                        {c: [{v: "17"}, {v: 605464}]},
                        {c: [{v: "11"}, {v: 190917}]},
                        {c: [{v: "Other"}, {v: 160264}]},
					]
				}
}' style="height:400px; width:100%;"></div>
</div>
<div class='col-md-6'>
<div google-chart chart='{
options: {
  title: "Repartition of customers per month",
  legend: {position: "none"},
  vAxis: { viewWindow: { min: 0, max: 1000000} }
},
type: "ColumnChart",
data: {
					"cols": [
						{id: "t", label: "Month", type: "string"},
						{id: "s", label: "Rows", type: "number"}
					], 
					"rows": [
                        {c: [{v: "2015-01"}, {v: 625457}]},
                        {c: [{v: "2015-02"}, {v: 627394}]},
                        {c: [{v: "2015-03"}, {v: 629209}]},
                        {c: [{v: "2015-04"}, {v: 630367}]},
                        {c: [{v: "2015-05"}, {v: 631957}]},
                        {c: [{v: "2015-06"}, {v: 632110}]},
                        {c: [{v: "2015-07"}, {v: 829817}]},
                        {c: [{v: "2015-08"}, {v: 843201}]},
                        {c: [{v: "2015-09"}, {v: 865440}]},
                        {c: [{v: "2015-10"}, {v: 892251}]},
                        {c: [{v: "2015-11"}, {v: 906109}]},
                        {c: [{v: "2015-12"}, {v: 912021}]},
                        {c: [{v: "2016-01"}, {v: 916269}]},
                        {c: [{v: "2016-02"}, {v: 920904}]},
                        {c: [{v: "2016-03"}, {v: 925076}]},
                        {c: [{v: "2016-04"}, {v: 928274}]},
                        {c: [{v: "2016-05"}, {v: 931453}]},
					]
				}
}' style="height:400px; width:100%;"></div>
</div>
</div>

Test set:

+ 930k unique customers
+ only one month history: 2016-06
+ but all customers are also in the train test

Our goal is to predict what a customer will buy in addition to what they already had at 2016-05-28.


#### How to train our models

Now that we understand the data a little better, we can think about how to train our models.

A few ideas:

+ once a customer purchases a product, how often does she cancel it ?
+ are there products more popular than others ?
+ We know if a customer is new or not (less than six months), so we can focus on their activity during those first months. It seems likely that they will buy more products at the beginning.
+ In a similar idea, we can track the number of months since subscription to the bank
+ We can create a new column of products bought & cancelled for each month.




# CLEANING DATA

The first step is to clean and prepare the data:

+ convert all values to numeric
+ reduce dimensionality of categorical features
+ fill `NA` values

#### Unused Customers

We start by removing the few customers that are not in the test set (approx. 1% of the training set), as they have many missing values.


#### Unused columns

We won't use columns that are either redundant or with too little useful information:

| Name | Description | Reason of unuse |
|----------|------------------|-------------------------|
| conyuemp | Spouse of employee index | Too few 'Yes' values (only 17)      |
| tipodom     | Address type                       | Only one value: 1                          |
| indresi       | Spain Residence index        | Redundant with pais_residencia   |
| cod_prov    | Province code (only for Spain residents) | Redundant with nomprov |
| antiguedad | Customer seniority (in months) | Redundant with fecha_dato - fecha_alta |
| ind_nuevo    | New customer Index | Redundant with fecha_dato - fecha_alta |
| indrel | Cancellation index | Too few '99' values (only 796) |
| ult_fec_cli_1t | Last date as primary customer | Too few values (only 796) |

#### Customer info

| Old Name | New Name  | Old Values | New Values | 
|----------------|-----------------|-----------------|------------------|
| ncodpers   | *same* | *Customer code* | *same* |
| sexo           |Sex          |  V or H               | Boolean; 1 for men |
| age             |Age        |  *Age of customer*                          |  *same*                    |
| indext        | Foreign_Birth  | S (Yes) or N (No) |   Boolean; 1 for foreign birth |
| ind_empleado  | Employee        |A active, B ex employed, F filial, N not employee, S passive  |  Boolean; 1 for employee/former employee |
| renta         | Income | *Gross income of the household*  | *same* |
| segmento   | Segment | 01 - VIP, 02 - Individuals 03 - college graduated |  1-2-3 |

<div class='row'>
<div class="col-md-8">
<div google-chart chart='{
options: { 
  title: "Overview",
  isStacked: "percent",
  legend: "bottom",
  colors: ["#3366CC", "#E7EDFB"]
}, 
type: "ColumnChart",
data: [
       ["What", "1", "0"],
       ["Sex", 6136554, 7354801],
       ["Foreign_Birth", 637731, 12853624],
       ["Employee", 8581, 13482774]
]}' style="height:450px; width:100%;"></div>
</div>
<div class="col-md-4">
<div google-chart chart='{
options: { 
  title: "Segment Split",
  isStacked: "percent",
  legend: "bottom",
  colors: ["#2C5AA0", "#3366CC", "#E7EDFB"]
  }, 
type: "ColumnChart",
data: [
       ["What", "1", "2", "3"],
       ["Segment", 554942, 7882565, 4901572],
]}' style="height:450px; width:100%;"></div>
</div>
</div>

Missing values:

| Name | Missing Values | Percentage Missing | Replaced by |
|----------|----------------------:|----------------------------:|------------------|
| ncodpers        | 0              | -  | - |
| Sex                   | 935          | - | most common value: 0 |
| Age                   | 865          | - | median age: 39              |
| Foreign_Birth  | 865           | - | most common value: 0 |
| Employee        | 865           | - | most common value: 0 |
| Income            | 2,719,619 |   20.2%  | - |
| Segment          | 152,276    | 1.1%       | - | 

Income & Segment have many missing values. Once the rest of the data is cleaned, we will use it to estimate the most likely values for these features.


#### Place of Residence

| Old Name | New Name  | Old Values | New Values | 
|----------------|-----------------|-----------------|------------------|
| pais_residencia | Res_Abroad | Country: ES, FR, etc. | Boolean; 1 if customer lives outside Spain |
| nomprov                | Res_CommName | Province Name | Autonomous Community |

Inconsistent values:

| Name | Values | Replaced by |
|----------|----------------------:|------------------|
| filled-in nomprov when pais_residencia not ES   | 100 | NA |
| missing  nomprov when pais_residencia is ES  | 92 | most common value: Madrid |
| row with no information | 865 | most common values: ES + Madrid | 

There are more than 50 [provinces](https://en.wikipedia.org/wiki/Provinces_of_Spain) in Spain, plus two autonomous cities in the coast of Africa. This is too large for One-Hot Encoding, so we will group them into their respective [autonomous communities](https://en.wikipedia.org/wiki/Autonomous_communities_of_Spain) plus `abroad` for customers living outside Spain. We then aggregate all communities with less than 300k rows, which gives us 13 one-hot columns for residence.

<div class='row'>
<div google-chart chart='{
options: {
  title: "Repartition of customer rows per autonomous community",
  legend: {position: "none"},
},
type: "ColumnChart",
data: {
					"cols": [
						{id: "t", label: "Community", type: "string"},
						{id: "s", label: "Rows", type: "number"}
					], 
					"rows": [
                        {c: [{v: "Madrid"}, {v: 4364390}]},
                        {c: [{v: "Andalusia"}, {v: 1820598}]},
                        {c: [{v: "Catalonia"}, {v: 1531229}]},
                        {c: [{v: "Valencia"}, {v: 1089949}]},
                        {c: [{v: "Galicia"}, {v: 870838}]},
                        {c: [{v: "Castile"}, {v: 774040}]},
                        {c: [{v: "LaMancha"}, {v: 533296}]},
                        {c: [{v: "Aragon"}, {v: 402061}]},
                        {c: [{v: "Murcia"}, {v: 394148}]},
                        {c: [{v: "Extremadura"}, {v: 320394}]},
                        {c: [{v: "Canary"}, {v: 303655}]},
                        {c: [{v: "Abroad"}, {v: 65589}]},
                        {c: [{v: "Other"}, {v: 1021168}]},
					]
				}
}' style="height:400px; width:100%;"></div>
</div>


#### Activity

| Old Name | New Name  | Old Values | New Values | 
|----------------|-----------------|-----------------|------------------|
| fecha_dato | Month        | 201x-xx-28  | from 1  (2015-01) to 17 (2016-06) |
| fecha_alta   | - | First month as customer | *used to compute seniority* |
| indrel_1mes | Customer | 1 (primary customer),<br>2 (co-owner ),<br>P (Potential),<br>3 (former primary),<br>4 (former co-owner) | Boolean; 1 if actual customer |
| tiprel_1mes | Active_Rel | A (active),<br>I (inactive),<br>P (former customer),<br>R (Potential) | Boolean; 1 if active relation| 
| ind_actividad _cliente | Activity | Boolean; 1 if active customer | Boolean; 1 if active customer|
| - | Seniority   | - | Computed; fecha_dato - fecha_alta |
| - | New_Customer   | - | Boolean; 1 if less than 6 months of Seniority |
| indfall | Deceased   | S (Yes) or N (No) | Boolean; 1 if deceased |

<div class='row'>
<div class='col-md-8'>
<div google-chart chart='{
options: { 
  title: "Overview",
  isStacked: "percent",
  legend: "bottom",
  colors: ["#3366CC", "#E7EDFB"]
  }, 
type: "ColumnChart",
data: [
       ["What", "1", "0"],
       ["Customer", 13372092, 4549],
       ["Active_Rel", 6117026, 7259615],
       ["Activity", 6179925, 7311430],
       ["New_Customer", 1103712, 12387643],
       ["Deceased", 31590, 13459765],
]}' style="height:450px; width:100%;"></div>
</div>
<div class='col-md-4'>
<div google-chart chart='{
options: { 
  title: "Activity vs Active_Rel",
  isStacked: "percent",
  legend: "bottom",
  colors: ["#3366CC", "#E7EDFB"]
  }, 
type: "ColumnChart",data: [
       ["What", "Active_Rel_0", "Active_Rel_1"],
       ["Activity_0", 6639024, 595023],
       ["Activity_1", 620591, 5522003],
]}' style="height:450px; width:100%;"></div>
</div>
</div>

Missing values:

| Name | Missing Values | Percentage Missing | Replaced by |
|----------|----------------------:|----------------------------:|------------------|
| Month        | 0              | -  | - |
| Customer  | 114714     | 0.85% | most common value: 1 |
| Active_Rel   | 114714     | 0.85% | Activity value (high correlation) |
| Activity        | 865           | - | most common value: 0 |
| Deceased    | 865           | - | most common value: 0 |
| Seniority     | 865           | - | most common value: 6 |
| New_Customer     | 865           | - | 1,  based on most common seniority |

<hr>

There are some unexplained discrepancies between the computed seniority and the `antiguedad` column; in most cases, the seniority is reduced by a few months.

<div class='row'>
<div google-chart chart='{
options: {
  title: "Gap between computed seniority and antiguedad",
  hAxis: { 
        slantedText: true, 
        slantedTextAngle: 30
    },
},
type: "PieChart",
data: {
					"cols": [
						{id: "t", label: "Gap", type: "string"},
						{id: "s", label: "Rows", type: "number"}
					], 
					"rows": [
                        {c: [{v: "No gap"}, {v: 8703332}]},
                        {c: [{v: "-1 to -6 months"}, {v: 3911873}]},
                        {c: [{v: "Others"}, {v: 875247}]},
					]
				}
}' style="height:400px; width:100%;"></div>
</div>

Possible additional steps:

+ Reactivation: when an account become active again
+ Reactivation during the last 6 months
+ check if properly inactive after cancellation


#### Channel

| Pos | Name  | Description  | Values  |
|:-----:|-----------|-------------------|------------|
| 16  | canal_entrada | channel used by the customer to join |   160+ different channels (non explicit values) |

160 differents channels are too many for one-hot encoding, especially considering the majority only have very low occurrences.

After cross-checking, no low-frequency channel is overly linked to any segment, which could have been a good reason to keep them. We will apply the following rules instead:

+ keep the five most frequent channels, with over 400k rows
+ aggregate all the others channels in groups based on their first two letters
+ aggregate the new groups that have less than 100k rows to the group `Others`

_Note: there is no missing value for the channels._

The final repartition in 10 groups:

<div>
<div google-chart chart='{
options: { 
  title: "Channels", 
  hAxis: { slantedText: true, slantedTextAngle: 30 }
}, type: "ColumnChart",
data: [
       ["Channel", "Rows"],
       ["KHE", 4029146],
       ["KAT", 3240977],
       ["KFC", 3072790],
       ["KH", 727959],
       ["KA", 699818],
       ["KHQ", 587215],
       ["KFA", 404198],
       ["KC", 127379],
       ["KF", 108680],
       ["Others", 493193],
]}' style="height:400px; width:100%;"></div>
</div>
</div>

#### Accounts

| Pos | Name  | Description  | Values  |
|:-----:|-----------|-------------------|------------|
| 24  | ind_ahor_fin_ult1     | Saving Account | 1 if the customer has an account |
| 25  | ind_aval_fin_ult1      | Guarantees | |
| 26  | ind_cco_fin_ult1      | Current Accounts | |
| 27  | ind_cder_fin_ult1     | Derivada Account | |
| 28  | ind_cno_fin_ult1      | Payroll Account | |
| 29  | ind_ctju_fin_ult1      | Junior Account | |
| 30  | ind_ctma_fin_ult1    | Más particular Account | |
| 31  | ind_ctop_fin_ult1     | particular Account | |
| 32  | ind_ctpp_fin_ult1     | particular Plus Account | |
| 33  | ind_deco_fin_ult1     | Short-term deposits | |
| 34  | ind_deme_fin_ult1    | Medium-term deposits | |
| 35  | ind_dela_fin_ult1      | Long-term deposits | |
| 36  | ind_ecue_fin_ult1     | e-account | |
| 37  | ind_fond_fin_ult1     | Funds | |
| 38  | ind_hip_fin_ult1       | Mortgage | |
| 39  | ind_plan_fin_ult1     | Pensions | |
| 40  | ind_pres_fin_ult1     | Loans | |
| 41  | ind_reca_fin_ult1     | Taxes | |
| 42  | ind_tjcr_fin_ult1      | Credit Card | |
| 43  | ind_valo_fin_ult1     | Securities | |
| 44  | ind_viv_fin_ult1           | Home Account | |
| 45  | ind_nomina_ult1         | Payroll | |
| 46  | ind_nom_pens_ult1     | Pensions | |
| 47  | ind_recibo_ult1           | Direct Debit | |

The vast majority of customers have only one account: the Current Account. 

The scoring method for the last month is 0 when the customer does not buy a new product, regardless of the predictions. We can create a default list of the most popular accounts, remove the ones each customer already have, and have our first prediction model.


<div class='row'>
<div google-chart chart='{
options: {
  title: "Popularity of each account",
  legend: {position: "none"},
  hAxis: { 
        slantedText: true, 
        slantedTextAngle: 30
    },
},
type: "ColumnChart",
data: {
					"cols": [
						{id: "t", label: "Account", type: "string"},
						{id: "s", label: "Rows", type: "number"}
					], 
					"rows": [
                        {c: [{v: "Current"}, {v: 8876279}]},
                        {c: [{v: "Particular"}, {v: 1750064}]},
                        {c: [{v: "Direct Debit"}, {v: 1733796}]},
                        {c: [{v: "E-account"}, {v: 1118638}]},
                        {c: [{v: "Payroll"}, {v: 1097126}]},
                        {c: [{v: "Pensions2"}, {v: 806426}]},
                        {c: [{v: "Payroll2"}, {v: 742816}]},
                        {c: [{v: "Taxes"}, {v: 702731}]},
                        {c: [{v: "Credit Card"}, {v: 603161}]},
                        {c: [{v: "Particular Plus"}, {v: 587279}]},
                        {c: [{v: "Long-Term"}, {v: 581449}]},
                        {c: [{v: "Others"}, {v: 1199122}]},                        
					]
				}
}' style="height:400px; width:100%;"></div>
</div>

<div class='row'>
<div google-chart chart='{
options: {
  title: "Accounts per customer",
  legend: {position: "none"},
  hAxis: { 
        slantedText: true, 
        slantedTextAngle: 30
    },
},
type: "ColumnChart",
data: {
					"cols": [
						{id: "t", label: "Accounts", type: "string"},
						{id: "s", label: "Rows", type: "number"}
					], 
					"rows": [
                        {c: [{v: "0"}, {v: 2506140}]},
                        {c: [{v: "1"}, {v: 7094106}]},
                        {c: [{v: "2"}, {v: 1909707}]},
                        {c: [{v: "3"}, {v: 764880}]},
                        {c: [{v: "4"}, {v: 447759}]},
                        {c: [{v: "5"}, {v: 288732}]},
                        {c: [{v: "6+"}, {v: 480031}]},                      
					]
				}
}' style="height:400px; width:100%;"></div>
</div>

#### Outcome & Segmentation

Outcome distribution:

<img src= 'https://sebastienplat.s3.amazonaws.com/22c6f0a38960140b9609a28932369b1f1480506080290' style="max-height:400px" class="img-responsive center-block"/>




# APPENDIX: TABLE FIELDS

tiprel_1mes          A          I    N       P      R
indrel_1mes                                          
1            6115990.0  7254848.0  NaN     NaN    NaN
2               1036.0      218.0  NaN     NaN    NaN
3                  NaN        NaN  NaN  3470.0    NaN
4                  NaN        NaN  NaN   268.0    NaN
P                  NaN        NaN  3.0     NaN  808.0


| Pos | Name                  | Description                                                                          | Values                                                                                         |
|-----|-----------------------|--------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| 0   | fecha_dato            | Month                                                                                | 201x-xx-28                                                                                     |
| 1   | ncodpers              | Customer code                                                                        |                                                                                                |
| 2   | ind_empleado          | Employee index                                                                       | A active, B ex employed, F filial, N not employee, S passive                                   |
| 3   | pais_residencia       | Country of residence                                                                 | ES, FR, etc. (99.3% ES)                                                                        |
| 4   | sexo                  | Sex                                                                                  | V or H                                                                                               |
| 5   | age                   | Age                                                                                  |                                                                                                |
| 6   | fecha_alta            | The date in which the customer became as the first holder of a contract in the bank  |                                                                                                |
| 7   | ind_nuevo             | New customer Index.                                                                  | 1 if the customer registered in the last 6 months.                                             |
| 8   | antiguedad            | Customer seniority (in months)                                                       |                                                                                                |
| 9   | indrel                |                                                                                      | 1 (First/Primary), 99 (Primary customer during the month but not at the end of the month)      |
| 10  | ult_fec_cli_1t        | Last date as primary customer (if he isn't at the end of the month)                  |                                                                                                |
| 11  | indrel_1mes           | Customer type at the beginning of the month                                          | 1 (First/Primary customer), 2 (co-owner ),P (Potential),3 (former primary), 4(former co-owner) |
| 12  | tiprel_1mes           | Customer relation type at the beginning of the month                                 | A (active), I (inactive), P (former customer),R (Potential)                                    |
| 13  | indresi               | Residence index (if the residence country is the same than the bank country)         | S (Yes) when pais_residencia = ES or N (No)                                                                              |
| 14  | indext                | Foreigner index (if the customer's birth country is different than the bank country) | S (Yes) or N (No)                                                                              |
| 15  | conyuemp              | Spouse of employee index                                                                         | 1 if the customer is spouse of an employee                                                      |
| 16  | canal_entrada         | channel used by the customer to join                                                 |   160+ different channels (non explicit values)                                                                                             |
| 17  | indfall               | Deceased index                                                                       | S (Yes) or N (No)                                                                              |
| 18  | tipodom               | Address type                                                                         | 1 (primary address) or no value                                                                            |
| 19  | cod_prov              | Province code (customer's address)                                                    | Only for Spain residents                                                                                          |
| 20  | nomprov               | Province name                                                                        | Only for Spain residents                                                                                               |
| 21  | ind_actividad_cliente | Activity index                                                                       | 1, active customer; 0, inactive customer                                                       |
| 22  | renta                 | Gross income of the household                                                        |                                                                                                |
| 23  | segmento              | segmentation                                                                         | 01 - VIP, 02 - Individuals 03 - college graduated                                              |
| 24  | ind_ahor_fin_ult1     | Saving Account                                                                       | 1 if the customer has an account                                                                                               |
| 25  | ind_aval_fin_ult1     | Guarantees                                                                           |                                                                                                |
| 26  | ind_cco_fin_ult1      | Current Accounts                                                                     |                                                                                                |
| 27  | ind_cder_fin_ult1     | Derivada Account                                                                     |                                                                                                |
| 28  | ind_cno_fin_ult1      | Payroll Account                                                                      |                                                                                                |
| 29  | ind_ctju_fin_ult1     | Junior Account                                                                       |                                                                                                |
| 30  | ind_ctma_fin_ult1     | Más particular Account                                                               |                                                                                                |
| 31  | ind_ctop_fin_ult1     | particular Account                                                                   |                                                                                                |
| 32  | ind_ctpp_fin_ult1     | particular Plus Account                                                              |                                                                                                |
| 33  | ind_deco_fin_ult1     | Short-term deposits                                                                  |                                                                                                |
| 34  | ind_deme_fin_ult1     | Medium-term deposits                                                                 |                                                                                                |
| 35  | ind_dela_fin_ult1     | Long-term deposits                                                                   |                                                                                                |
| 36  | ind_ecue_fin_ult1     | e-account                                                                            |                                                                                                |
| 37  | ind_fond_fin_ult1     | Funds                                                                                |                                                                                                |
| 38  | ind_hip_fin_ult1      | Mortgage                                                                             |                                                                                                |
| 39  | ind_plan_fin_ult1     | Pensions                                                                             |                                                                                                |
| 40  | ind_pres_fin_ult1     | Loans                                                                                |                                                                                                |
| 41  | ind_reca_fin_ult1     | Taxes                                                                                |                                                                                                |
| 42  | ind_tjcr_fin_ult1     | Credit Card                                                                          |                                                                                                |
| 43  | ind_valo_fin_ult1     | Securities                                                                           |                                                                                                |
| 44  | ind_viv_fin_ult1      | Home Account                                                                         |                                                                                                |
| 45  | ind_nomina_ult1       | Payroll                                                                              |                                                                                                |
| 46  | ind_nom_pens_ult1     | Pensions                                                                             |                                                                                                |
| 47  | ind_recibo_ult1       | Direct Debit                                                                         |                                                                                                |