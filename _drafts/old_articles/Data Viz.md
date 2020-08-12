[Matplotlib](http://matplotlib.org/) is a very powerful plotting library. Its many parameters allow you to create presentation-ready graphs in no time. 

# MATPLOTLIB - PLOT CREATION & LAYOUT

In Matplotlib, a graph is divided in two parts:

+ the `figure`, which is the canvas (or container) for your plots
+ the `axes`, which are your actual plots

A `figure` can have one or several `axes`, ie. one or several plots. Some properties are linked to the figure (the figure size, for example) and some are linked to each axis (the type of graph: bar, lines, etc.).

The [complete Axes class API](http://matplotlib.org/api/axes_api.html) covers all properties and methods of the `axes` class.

You have essentially three ways of creating a graph:

+ **default**: when you use `plt.plot(x, y)` for example
+ **manually**: when you want to control precisely where your axes are located inside the figure
+ **automatically**: when you want to use a grid layout with rows and columns

The default is a particular case of the automatic method with only one graph.




#### Default 

Once figures and axes have been created, they can be accessed with:
+ figure: `plt.gcf()`  (stands for "get current figure")
+ axis: `plt.gca()`  (stands for "get current axis")

This can be used to set properties values, like so: `plt.gca().set_xlim([Xmin, Xmax])`.

#### Manual layout

You create a manual layout with the following steps:

1. Create new instance of the Figure class: blank canvas
2. Create a new instance of Axes class. Four params as % of the height/width:
     + distance from left / distance from bottom
     + width / height
3. Use each Axes instance for plotting 

Example:

```python
# import relevant modules
import matplotlib.pyplot as plt
import numpy as np

# source data
x = np.linspace(0, 5, 11)
y = x ** 2

# figure canvas
fig = plt.figure()

# first plot (large one)
axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
axes1.plot(x, y)
axes1.set_title('Manual subplots')

# second plot (small one inside)
axes2 = fig.add_axes([0.2,0.5,0.3,0.3])
axes2.plot(y, x)
```

<img class="center-block" src="https://sebastienplat.s3.amazonaws.com/173ceae08eb21d74749ff672cb91ae711492014948842"/>




<br>
#### Automatic layout

The `subplots` method creates a layout automatically, based on the `nrows` and `ncols` parameters. `figsize` accepts a tuple (width, height) in inches. Its default is (6.4, 4.8).

The `fig.tight_layout()` method prevents overlapping of subplots. It should always be applied in the end, because it won't consider titles set afterwards. 

It is best coupled with `fig.subplots_adjust(top=0.xx)` to display the title properly (no overlapping w/ top subplots). You can adjust `0.xx` to fit your needs.

Example:

```
# toy data
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

# create blank canvas
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8,3))

# add plots
ax1.plot(x, y)
ax2.scatter(x, y)

# add titles
fig.suptitle("This is the figure title", fontsize=18)
ax1.set_title('This is the ax1 title')

# tidy up layout
fig.tight_layout()
fig.subplots_adjust(top=0.75)
```

<img class="center-block" src="https://sebastienplat.s3.amazonaws.com/3a668db304a2cc40f44abe093ca2bd731492016648338">

<br>
See also the [subplots API](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplots).




#### Subplots w/  loops

It is also possible to create subplots with loops. This can be useful to create facet grids.

Example:

```
# toy data
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

# figure parameters
plot_shape = (2, 3)
figsize = (8,3)

# figure canvas
fig = plt.figure(figsize=figsize)

# add plots
for idx in range(6):
    ax1 = fig.add_subplot(plot_shape[0], plot_shape[1], idx+1)
    ax1.plot(x, y)
    
# tidy up layout
fig.tight_layout()
```

<img class="center-block" src="https://sebastienplat.s3.amazonaws.com/8dbbc56f99e624b16e399c7f422d80ca1492017103279"/>




<br>
#### Complex Grid Layout

You can also create complex grids layouts, with plots spanning multiple rows and columns.

Example:

```
# figure size
fig = plt.figure(figsize=(16,8))

# layout
ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
ax2 = plt.subplot2grid((3,3), (1,0), colspan=2)
ax3 = plt.subplot2grid((3,3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3,3), (2, 0))
ax5 = plt.subplot2grid((3,3), (2, 1))
```

<img class="center-block" src="https://sebastienplat.s3.amazonaws.com/859e5cb463500161090f5f4a9c537dda1492017421248"/>

<br>
See also:

+ custom subplot location with [gridspec](http://matplotlib.org/users/gridspec.html)
+ complete [layout guide](http://matplotlib.org/users/tight_layout_guide.html)




#### Spacing between subplots

You saw a few examples of tight layout, but it can be fully customized to set the distance between subplots and the padding around the graph area.

Example of syntax:

```
# automatic layout
fig.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
```

Manual adjustments are possible. All values are expressed as percentage of the horizontal or vertical axes.

```
# manual adjustments
fig.subplots_adjust(
    left=None, right=None, 
    bottom=None, top=None,
    wspace=None, hspace=None
)
```

See also:

+ [tight layout API](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.tight_layout)
+ [subplots\_adjust API](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots_adjust.html)




# MATPLOTLIB - AXIS FORMATTING

The [complete Axis class API](http://matplotlib.org/api/axis_api.html) covers all properties and methods of the `axis` class.

#### Ticks & Grid

Matplotlib has major and minor ticks for both axes, associated with labels. It is possible to fully customize their appearance.

You will see the formatting of labels in the next section. 

See also the [API](http://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.tick_params.html).



```
# remove minor ticks
ax.minorticks_off()

# Add autoscaling minor ticks
ax.minorticks_on()
```

```
# labels + ticks w/ color
ax.set_xlabel('my_label', color='my_color')
ax.tick_params(axis='x', colors='my_color')
```

```
# no tick labels
ax.set_xticklabels([]) 

# hide axis grid
ax.xaxis.grid(False)
 
 # hide axis
 ax.xaxis.set_visible(False)
 
# hide x ticks for all subplots except the last x ones
plt.setp([a.get_xticklabels() for a in fig.axes[:-x]], visible=False)
```




#### Tick Labels

You can format the numbers displayed in the axes with the  `set_major_formatter` and `set_minor_formatter` methods. The `ticker` module makes it very easy.

There are many formatting examples in the [Python documentation](https://docs.python.org/3/library/string.html#formatexamples), but two very common formats are thousands and percentages:

```
# import ticker module
import matplotlib.ticker as tkr

# comma for thousands separator
ax.xaxis.set_major_formatter(
    tkr.FuncFormatter(lambda x, p: "{:,}".format(int(x))))
    
# number as %
ax.xaxis.set_major_formatter(
    tkr.FuncFormatter(lambda x, p: "{:.1%}".format(x)))
```

You can also remove the labels by using the `NullFormatter` method:

```python
# remove labels for minor ticks
ax1.xaxis.set_minor_formatter(tkr.NullFormatter())
```

See also the [ticker API](http://matplotlib.org/api/ticker_api.html) to manually adjust ticks positions.

```
ax.xaxis.set_major_locator(...)
```

+ [`tkr.LinearLocator(x)`](https://matplotlib.org/api/ticker_api.html#matplotlib.ticker.LinearLocator) : x ticks from x\_min to x\_max
+ [`tkr.FixedLocator(locs)`](https://matplotlib.org/api/ticker_api.html#matplotlib.ticker.FixedLocator) : array of tick values. Can be used with`range(min, max, step)`

And also for the xaxis:

+ [`tkr.IndexLocator(x, offset)`](https://matplotlib.org/api/ticker_api.html#matplotlib.ticker.IndexLocator) : ticks every x values, starting from offset (offset has no default so you must indicate it).

_Note: labels get automatically centered on ticks, which is not ideal for bar charts. _

Labels can overlap when they are too long. You can use the [autofmt_xdate](https://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure.autofmt_xdate) method to rotate the labels. It is very useful for dates, hence the name, but can be used for all cases:

```
# autofmt_xdate method with default values
fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
```

Please note that it is a `figure` method. It will not work if used on subplots.




#### Tick Labels for Time Series

When plotting time series, it is often interesting to mark weeks, months or years. You can do it by using the `dates` module.

You might need to explicitly tell Matplotlib to treat the data as dates:

```
# treat x data as dates
ax.xaxis_date()
```

The following example from the [Matplotlib documentation](http://matplotlib.org/examples/pylab_examples/finance_demo.html) illustrates how powerful the module can be: historical quotes for Yahoo.

```
### Get Yahoo quotes

from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc

# args for quotes_historical_yahoo
date1 = (2004, 2, 1)
date2 = (2004, 4, 12)

# get quotes
quotes = quotes_historical_yahoo_ohlc('INTC', date1, date2)
if len(quotes) == 0:
    raise SystemExit
```

The quotes can then be plotted. You create two identical graphs.

```
### Plot Yahoo quotes

import matplotlib.pyplot as plt

# blank canvas
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))

# plot_day_summary
candlestick_ohlc(ax1, quotes, width=0.6)
candlestick_ohlc(ax2, quotes, width=0.6)

# x-axis as dates
ax1.xaxis_date()
ax2.xaxis_date()

# autorotate labels
fig.autofmt_xdate()
```

Then, you update the labels of the graph on the right, to have ticks every monday. The graph on the left keeps its default formatting.

```
### Format dates

from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY

# proper formatting 
mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
alldays = DayLocator()                  # minor ticks on the days
weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
#dayFormatter = DateFormatter('%d')      # e.g., 12

# apply formatting to the graph on the right
ax2.xaxis.set_major_locator(mondays)
ax2.xaxis.set_minor_locator(alldays)
ax2.xaxis.set_major_formatter(weekFormatter)
#ax2.xaxis.set_minor_formatter(dayFormatter)

plt.show()
```

<img class="center-block" src="https://sebastienplat.s3.amazonaws.com/a171f84cbf1953ac8337f8eec846a5071492027125835"/>

The `DateFormatter` method uses the `strftime` date format. Its options are listed at [strftime.org](http://strftime.org/).




#### Limits

The limits of a subplot are set automatically. You can override the default with the `set_xlim` and `set_ylim` methods:

```
# limits
ax.set_xlim([min, max]) # min=None or max=None for inferred value
```

Information outside these limits are not displayed. 

You can use the `set_xbound` and `set_ybound` methods instead for more flexibility. The difference is explained in this [SO question](http://stackoverflow.com/questions/11459672/in-matplotlib-what-is-the-difference-betweent-set-xlim-and-set-xbound).




#### Dual axes

```
# we suppose ax1 has already been created 
# and color was set to "cornflowerblue"
ax1_xticks = ax1.get_xticks()
ax2 = ax1.twinx()
ax2.plot(ax1_xticks, my_y_values, color="maroon")

# make the y-axis label, ticks and tick labels match the line colors    
ax1.set_ylabel('number of users', color='cornflowerblue')
ax1.tick_params('y', colors='cornflowerblue')
ax2.set_ylabel('conversion rate', color='maroon')
ax2.tick_params('y', colors='maroon')
```




#### Broken Axis

You can combine dual axes with cleverly set limits to create a broken axis. This is useful to plot outliers that would otherwise hide the details of the rest of the graph.

See also [Broken axis](http://matplotlib.org/examples/pylab_examples/broken_axis.html).

```
# figure layout
f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

# plot the same data on both axes
ax1.plot(pts)
ax2.plot(pts)

# zoom-in / limit the view to different portions of the data
ax1.set_ylim(.78, 1.)  # outliers only
ax2.set_ylim(0, .22)  # most of the data

# hide the spines between ax and ax2
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop='off')  # don't put tick labels at the top
ax2.xaxis.tick_bottom()
```




# MATPLOTLIB - OTHERS

#### Titles

```
# figure title
fig.suptitle("This is the figure title", fontsize=18)

# axis title
ax.set_title('my_title')
```

#### Horizontal and Vertical Lines

```
# vertical line
ax.axvline(x_pos)

# horizontal line
ax.axhline(y_pos)
```

See alo the docs for [`axvline`](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.axvline) and  [`axhline`](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.axhline).

#### Legend & Caption

##### Legend

See also :

+ [legend API](http://matplotlib.org/api/legend_api.html)
+ legend in [arbitrary position](http://matplotlib.org/api/legend_api.html#matplotlib.legend.Legend.set_bbox_to_anchor)
+ [legend guide](http://matplotlib.org/users/legend_guide.html)


The example below shows how to use a common legend for all subplots of a graph (two subplots in the example):

```
# legend
ax.legend(loc="lower right")
ax.legend().set_visible(True)

# remove subplots legend
ax1.legend_ = None
ax2.legend_ = None

# add space below the two subplots
fig.subplots_adjust(bottom=0.2, wspace=0.3)

# get legend from the first subplot
h, l = ax1.get_legend_handles_labels()

# apply the values to the graph legend
fig.legend(h, l, loc='lower center', ncol=3)
```

Legend placement:

```
ax.legend(bbox_to_anchor=(left, top))
```

The `left` and `top` values indicate the position of the left & top side of the Legend Box in relation to the graph area. They are expressed in percentage:

+ (0, 0): the legend is at the bottom left of the graph
+ (1, 0): the left of the Legend is at 100% of the graph width, i.e. outside
+ (0, 1): the top of the Legend is at 100% of the graph width, i.e. at the top

##### Caption

See also: 

+ [text API](http://matplotlib.org/api/text_api.html#matplotlib.text.Text)
+ autowrap [demo](http://matplotlib.org/examples/text_labels_and_annotations/autowrap_demo.html).

To add a caption at the bottom of a graph:

```
# x, y: text box coordinates
# ha: horizontal alignment of the text box 
# (point of the text box used for x, y)
# multialignment: text alignment inside the text box
txt = my_text
fig.subplots_adjust(bottom=0.4)
fig.text(x=0.5, y=0, txt, ha='center', multialignment='left', wrap=True)
```

Example of caption for Fig. titles:

```
txt1 = "Fig. x: Fig title"
txt2= '''
  Line1.
  Line2.
'''

fig.subplots_adjust(bottom=0.25)
fig.text(0.2, 0.12, txt1, weight="demibold", wrap=True)
fig.text(0.2, 0, txt2, wrap=True)
```

Depending on your text length, you will probably need to adjust the parameters.

_Note: Autowrap does not work in Jupyter notebooks. You can manually break the text by using linebreaks or triple quotes._


<br>


#### Plots Formatting

Matplotlib offers great control over figures layout when using the Figure and Axes classes.

##### Color & Style

See also:

+ [Lines API](http://matplotlib.org/api/lines_api.html)
+ [Markers API](http://matplotlib.org/api/markers_api.html)
+ [Named Colors](https://i.stack.imgur.com/fMx2j.png)

```python
# color & style
ax.plot(x, y, 
  color="#FF8C00",
  lw=2, # linewidth; default 1
  ls="-", # linestyle; default "-"
  marker="o", # ="None" to remove them
  markersize=8,
  markerfacecolor="purple",
  markeredgewidth=2,
  markeredgecolor="blue",
  alpha=0.5) # affects line + marker
        
# custom lines styles - 4 in this example
# should be placed before the plot function
# otherwise the style will not be applied
from cycler import cycler

ax1.set_prop_cycle(cycler('color', ['blue', 'grey', 'grey', 'grey']) +
                   cycler('lw', [2, 1, 1, 1]) + 
                   cycler('alpha', [1, 0.5, 0.5, 0.5]))
                   
# bar - remove edge line 
bar(..., edgecolor = "none")
```

##### Z Positioning

```
# use zorder to move lines behind the plot
plt.axhline(y=200, color="lightgray", zorder=1)
```

##### Fill Between Lines

```
ax.fill_between(x, y1, y2)
```

See also these Matlotlib [examples](https://matplotlib.org/gallery/lines_bars_and_markers/fill_between_demo.html).

<br>


#### Zebra Stripes

See also:

+ [bar API](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.bar)
+ [zebra stripes](http://stackoverflow.com/questions/2815455/matplotlib-zebra-stripe-a-figures-background-color)
+ [custom rectangles](http://matthiaseisen.com/pp/patterns/p0203/).

#### Example

```
# useful libraries
import matplotlib.pyplot as plt
import numpy as np

# source data
x = np.linspace(0, 5, 11)
y = x ** 2

# blank canvas
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7,7))
fig.suptitle("Automatically divided subplots - m*n plots", fontsize=20)

# iterate through plots
for idx, ax in enumerate(axes.flatten()):
  ax.set_title(idx)
  if idx < 3:
    ax.plot(x, y, color="green")

# specific parameters for the last plot    
axes[1,1].plot(y, x, label="x = f(y)", 
               color="#FF8C00",
               lw=2, # linewidth; default 1
               ls="-", # linestyle; default "-"
               marker="o",
               markersize=8,
               markerfacecolor="purple",
               markeredgewidth=2,
               markeredgecolor="blue",
               alpha=0.5) # affects line + marker

axes[1,1].legend(loc="lower right")
axes[1,1].legend().set_visible(True)

axes[1,1].set_xlim([0,30])
axes[1,1].set_ylim([0,10])

# clean layout
fig.tight_layout()
fig.subplots_adjust(top=0.88)

# x axis label angle
fig.xticks(rotation=0)
ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)

# save to file
fig.savefig("example.png", dpi=200)
```

![example.png](https://sebastienplat.s3.amazonaws.com/0446ebdc16030ec32a1e84d37cd0d9871485340957997)



#### Advices for Better Figures

+ [Advices for better figures](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833)




# SEABORN

[Seaborn](http://seaborn.pydata.org/index.html) is a library that add functionalities to Matplotlib, including useful statistical plots and a nicer output.

Some plots automatically create a new Matplotlib `Figure` instance, while others are added to the existing one.

_Note: all the examples are from the 'tips' dataset from seaborn. Matplotlib is needed to have more control over complex layouts._

```
# import useful libraries
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
```


#### Distribution plots

`distplot()` shows the univariate distribution of observations, with or without its [Kernel Density Estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation). 

```
sns.distplot(tips['total_bill'], kde=False, bins=30)
```

<img text="distplot" style="max-width: 300px;" src="https://sebastienplat.s3.amazonaws.com/823c1571bb3074028b40bd2747df90081485342163686"/>


`jointplot()` combines two distplots for bivariate data. Its `kind` parameters accepts a few values: `scatter`, `reg`, `resid`, `kde`, `hex`. 

`resid` returns the residual plot of the linear regression.

`size` indicates the total height & width of the graph. `ratio` is the relative size of the joint plot compared to the univariate plots height.

```
sns.jointplot(x='total_bill', y='tip', data=tips, kind='reg', size=5, ratio=6)
```

<img text="jointplot" style="max-width: 300px;" src="https://sebastienplat.s3.amazonaws.com/4e83691a337f8eab8e2a45bb1fe2517c1485344059325"/>


#### Complex plots

`order` specifies the display order of categorical data.

```
# figure canvas
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

# plots
sex_order = ['Male', 'Female']
sns.countplot(tips['day'], hue=tips['time'], ax=ax1)
sns.countplot(tips['sex'], hue=tips['time'], order=sex_order, ax=ax2)

# use same axes for both plots
ax1.set_ylim((None, 140))

# create common legend for both plots
ax1.legend_ = None
ax2.legend_ = None

fig.subplots_adjust(bottom=0.2, wspace=0.3)

h, l = ax1.get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3)
```

![countplots_complex.png](https://sebastienplat.s3.amazonaws.com/c9a706c9e6b858cec58f7fab57ecb98b1485344716157)

We can create a grid plot defined by features values (see [API](http://seaborn.pydata.org/generated/seaborn.FacetGrid.html#seaborn.FacetGrid)):

```
# create facet grid
g = sns.FacetGrid(df, row="feature1", col="feature2", hue="feature3")

# example with scatter plots
g = g.map(plt.scatter, "feature4", "feature5")

# example with histograms
bins = 10
g = g.map(plt.hist, "feature4", bins=bins, color="cornflowerblue")

# format subplots axes
for ax in g.axes.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.xaxis.set_major_formatter(
        tkr.FuncFormatter(lambda x, p: "{:}k".format(int(x/1000))))
```




# PANDAS

See also:

+ the [docs](http://pandas.pydata.org/pandas-docs/stable/visualization.html)
+ [facet plots](http://stackoverflow.com/questions/29786227/how-do-i-plot-facet-plots-in-pandas)
+ [custom facets in seaborn](http://seaborn.pydata.org/generated/seaborn.FacetGrid.html)

See also this [tutorial](https://plot.ly/python/big-data-analytics-with-pandas-and-sqlite/) about Pandas with SQLite.
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,4))
my_df.plot.hist(x=col_name_1, y=col_name_2, ax=ax, legend=None)
```




# GEO PLOTTING - FOLIUM

Install instruction can be found in [folium github page](https://github.com/python-visualization/folium).
