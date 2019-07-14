#Given: Type of TI and the relevant data for that particular TI
#Output: A Strength Estimate for the TI on a range of -10 to +10
#Meanwhile: Plot bhi kar lo us ko
#data format : {'Date':np.array, 'Open':np.array,  'High':np.array 'Low':np.array,  'Close':np.array }
import talib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go 
import plotly.tools as tools
import plotly.io as pio
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


def generate_freq_dist(data):
	ret = plt.hist(data,bins=20,density=True,stacked = True)
	# print (ret)
	# plt.show()
	return ret

def get_strength_EMA(data):
	#Exponential Moving Average applied on closing prices. Only default value of timeperiod used for now.
	op = talib.EMA(data['Close'])

	#plot
	list_traces = []
	list_traces.append(go.Scatter(x=data['Date'], y= op, name = 'momentum histogram'))
	list_traces.append(go.Candlestick(
		x=data['Date'],
		open = data['Open'],
		high = data['High'], 
		low = data['Low'], 
		close = data['Close'], 
		)
	)
	# list_traces.append(go.Candlestick(
	# 	x=data['Date'][450:],
	# 	open = data['Open'][450:],
	# 	high = data['High'][450:], 
	# 	low = data['Low'][450:], 
	# 	close = data['Close'][450:],
	# 	increasing = dict(
	# 		fillcolor = 'blue'),
	# 	decreasing = dict(
	# 		fillcolor = 'white')
	# 	))
	fig = go.Figure(data = list_traces)
	plot(fig)

	op = np.array(op)
	finite_diff = np.diff(op)
	print (finite_diff)
	# score = finite_diff[-1] * 10 / max(abs(np.nanmax(finite_diff)), abs(np.nanmin(finite_diff)))
	score = finite_diff[-1]
	if score > 10:
		score = 10
	elif score < -10:
		score = -10
	return score,fig

def get_strength_DEMA(data):
	return get_strength_EMA(data)

def get_strength_WILLR(data):
	#WILLR = (Highest Hgh - Close)/(Highest High - Lowest Lowest)
	op = talib.WILLR(data['High'],data['Low'], data['Close'])

	#plot
	list_traces=[]
	list_traces.append(go.Scatter(x=data['Date'], y= op, name = 'Williams %R'))
	fig = go.Figure(data=list_traces)
	plot(fig)

	
	#+5 for WILLR < -80 
	#-5 for WILLR > -20
	#+5 for both price and indicator going up
	#-5 for both price and indicator going down
	score = 0
	closing_price = np.array(data['Close'])
	op = np.array(op)
	if (closing_price[-1] < -80):
		score += 5
	elif(closing_price[-1] > -20):
		score += -5
	if (closing_price[-1] - closing_price[-2] > 0):
		if(op[-1] - op[-2] > 0):
			score += 5
	elif(closing_price[-1] - closing_price[-2] < 0):
		if(op[-1] - op[-2] < 0):
			score += -5

	return score,fig

def get_strength_RSI(data):
	'''
	1)if current price is on the rise, we look for most recent maxima and last two minimas. 
		if the most recent maxima and the current value form a +ve slope, Add +5 to the score, 
		if the last two minimas form a +ve slope, add +5
	2)
	'''
def get_strength_ADX(data):
	op = np.array(talib.ADX(data['High'], data['Low'], data['Close']))
	plus_DI = np.array(talib.PLUS_DI(data['High'], data['Low'], data['Close']))
	minus_DI = np.array(talib.MINUS_DI(data['High'], data['Low'], data['Close']))
	
	#plots
	list_traces = []
	list_traces.append(go.Scatter(x=data['Date'], y=op, name= 'ADX'))
	list_traces.append(go.Scatter(x=data['Date'], y=plus_DI, name= '+DI'))
	list_traces.append(go.Scatter(x=data['Date'], y=minus_DI, name= '-DI'))
	fig = go.Figure(data=list_traces)
	plot(fig)

	#score calculation
	score=0
	if (op[-1]>20):
		if(plus_DI[-1] > minus_DI[-1]):
			score = (op[-1]-20)*1.5
			if score>10:
				score = 10
		else:
			score = (op[-1]-20)*-1.5
			if score<-10:
				score = -10
	return score,fig

def get_strength_MACD(data):
	#we need only the close data for MACD analysis
	op = talib.MACD(data['Close'])
	hist = np.array(op[2]);

	#Get the MACD plot. "Intend to later: add the pricing chart above the macd chart"
	list_traces=[]
	list_traces.append(go.Scatter(x=data['Date'], y= op[0], name = 'macd'))
	list_traces.append(go.Scatter(x=data['Date'], y= op[1], name = 'signal line'))
	list_traces.append(go.Bar(x=data['Date'], y= op[2], name = 'momentum histogram'))
	fig = go.Figure(data=list_traces)

	plot(fig)

	# print (data['Date'][np.nanargmax(np.array(op[2]))])
	minm =  np.nanmin(hist)
	maxm =  np.nanmax(hist)
	
	#separating the positive and negative values
	#from the histogram obtained from the MACD calculation
	positives = []
	negatives = []
	for t in hist:
		if t>0 and not np.isnan(t):
			positives.append(t)
		elif not np.isnan(t):
			negatives.append(t)
	positives = np.array(positives)
	negatives = np.array(negatives)
	print (len(positives),len(negatives))

	positive_freq_dist = generate_freq_dist(positives)
	negative_freq_dist = generate_freq_dist(negatives)

	for i in range(len(positive_freq_dist[0])):
		if(positive_freq_dist[0][i]>0.015) and i!=0:
			maxm = positive_freq_dist[1][i]

	for i in range(len(negative_freq_dist[0])):
		if(negative_freq_dist[0][-1 + -1*i] > 0.015):
			minm = negative_freq_dist[1][-1 + -1*i]


	if hist[-1]>=0:
		current_score = hist[-1]*10/maxm
		if current_score>10:
			current_score = 10
	else:
		current_score = hist[-1]*-10/minm
		if current_score <-10:
			current_score = -10

	print ('curr score = ' + str(current_score))
	return current_score,fig


def get_strength_BBANDS(data):

	op = talib.BBANDS(data['Close'])
	tp = (data['High'] + data['Low'] + data['Close'])/3.0

	list_traces=[]
	list_traces.append(go.Scatter(x=data['Date'], y= op[0], name = 'macd'))
	list_traces.append(go.Scatter(x=data['Date'], y= op[1], name = 'signal line'))
	list_traces.append(go.Scatter(x=data['Date'], y= op[2], name = 'momentum histogram'))
	list_traces.append(go.Candlestick(x=data['Date'],open = data['Open'], high = data['High'], low = data['Low'], close = data['Close']))
	fig = go.Figure(data=list_traces)
	plot(fig)
	tp = np.array(tp)
	op = np.array(op)
	strength = 0
	if tp[-1] > op[1][-1]:
		strength =  (op[1][-1] - tp[-1])*10/(op[0][-1] - op[1][-1])
	else:
		strength = (tp[-1] - op[1][-1])*10/(op[2][-1] - op[1][-1])
	print (strength)
	return strength,fig			


def plot_candlestick(stockData):
	ohlc=stockData.getOhlc()
	trace_cs = go.Ohlc(x=ohlc['Date'],
	                open=ohlc['Open'],
	                high=ohlc['High'],
	                low=ohlc['Low'],
	                close=ohlc['Close'], name='Candlestick Pattern')

	data = [trace_cs]
	fig = go.Figure(data=data)
	plot(fig)

def get_ti_strength(ti_name,data):
	# try:
	score,fig  = eval('get_strength_' + ti_name)(data)
	print (score)
	# plot(fig,output_type = 'div', filename = temp_div)
	# plot_candlestick(data)
	return score,plot(fig,
     include_plotlyjs=False,
     output_type='div')
	# except:
	# 	print (e)
	# 	print ('Invalid Technical Indicator')


# data = pd.read_csv('TCS.BO.csv',usecols=['Date','Open', 'High', 'Low', 'Close']);
# # print (data)
# get_ti_strength('EMA',data)