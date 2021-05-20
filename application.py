from flask import Flask, render_template, request, redirect, url_for
import mysql.connector
from datetime import datetime
import os
import pandas as pd
import numpy as np
import mysql.connector
import sys
import json
import matplotlib.pyplot as plt
sys.path.append(".")
from calculations import *
from optimization import *
import pathlib

app = Flask(__name__)
# Upload folder
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

UploadedFileDF = []
UploadedFileStockName = []
PickedStock = [] #save UploadedFilePath and UploadedFileStockName index
OptimizationResult = []

@app.route('/')
@app.route('/home', methods=['GET', 'POST'])
def home():
	NewestTicker = checkTheNewestTicker()
	NewestPick = checkTheNewestPick()
	return render_template("home.html", newestTicker = NewestTicker , newestPick = NewestPick, lenNewestPick = len(NewestPick) )	

@app.route('/portfolio')
def portfolio():
	NewestTicker = checkTheNewestTicker()
	NewestPick = checkTheNewestPick()
	return render_template("portfolio.html", newestTicker = NewestTicker , newestPick = NewestPick, lenNewestPick = len(NewestPick) )		

@app.route("/uploadFile", methods = ['POST'])
def uploadFile():
	# get the uploaded file
	uploaded_file = request.files['myFile']
	if uploaded_file.filename != '':
		read_file = pd.read_csv(uploaded_file, parse_dates=["Date"])
		setUploadedFileDFAndStockName(read_file,read_file["Ticker"][0])
	return redirect(url_for('home'))

def setUploadedFileDFAndStockName(path,name):
	UploadedFileDF.append(path)
	UploadedFileStockName.append(name)

def getUploadedFileDF():
	return UploadedFileDF

def getUploadedFileStockName():
	return UploadedFileStockName

def setPickedStock(data):
	PickedStock.append(data)

def getPickedStock():
	return PickedStock

def pickedStockData(i): #enter PickedStock index return PickedStockData
	target = getPickedStock()
	target = target[i][1]
	fileStockName = getUploadedFileStockName()
	targetDFs = getUploadedFileDF()
	targetPathList = []
	for name in target:
		targetPathList.append(targetDFs[fileStockName.index(name)]) 
	
	return targetPathList

def checkTheNewestTicker():
	myresult = getUploadedFileStockName()
	return myresult

def checkTheNewestPick():
	myresult = getPickedStock()
	return myresult

@app.route('/pickStock', methods = ['POST'])
def pickStock():
	pickedTicker = request.form.getlist('Ticker')
	allStockName = getUploadedFileStockName()
	pickStock = []
	stockName = []
	weights = []
	startDay = []
	endDay = []
	stock_return = []
	risk = []
	expdecte_returns =""

	for i in pickedTicker:
		stockName.append(allStockName[allStockName.index(i)])

	pickStock.append(weights)
	pickStock.append(stockName)	
	pickStock.append(startDay)
	pickStock.append(endDay)
	pickStock.append(stock_return)
	pickStock.append(risk)
	pickStock.append(risk)

	setPickedStock(pickStock)
	#print(getPickedStock())
	return redirect(url_for('home'))

@app.route('/updatePorfolio')
@app.route('/updatePorfolio', methods = ['POST'])
def updatePorfolio():
	
	weights = OptimizationResult[0]
	tickers = OptimizationResult[1]	
	startDay = OptimizationResult[2]	
	endDay = OptimizationResult[3]	
	stock_return = OptimizationResult[4]	
	risk = OptimizationResult[5]
	portfoloIndex = OptimizationResult[6]
	
	risk = [ '%.3f' % elem for elem in risk ]
	weights = [ '%.3f' % elem for elem in weights ]
	stock_return = [ '%.3f' % elem for elem in stock_return ]

	new_porfolio = []

	new_porfolio.append(weights)
	new_porfolio.append(tickers)	
	new_porfolio.append(startDay)
	new_porfolio.append(endDay)
	new_porfolio.append(stock_return)	
	new_porfolio.append(risk)

	print(PickedStock[portfoloIndex])
	print(new_porfolio)

	PickedStock[portfoloIndex] = new_porfolio

	return redirect(url_for('home'))

@app.route('/estimation', methods=['GET','POST'])
def estimation():
	pickedStock = request.form.get('pickedStock1') # take portfolio data
	try:
		data = pickedStockData(int(float(pickedStock)))
	except:		
		return render_template("noPickPortfolio.html")

	print(data)	# example of how to take data
	print(data[0]["Ticker"][0]) # example of how to take data
	print(data[0]["Close"]) # example of how to take data
    
    
    
	df = pd.DataFrame()
    
	l_data = len(data)
    
	tickers = []
	for i in range(l_data):
		df['Date'] = pd.to_datetime(data[0]['Date'],format='%Y%m%d')
		df[data[i]["Ticker"][0]] = data[i]["Close"]
		tickers.append(data[i]["Ticker"][0])
	print(df)

	#df.set_index('Date', inplace=True)
	#df.info()

#	print(df.info())

	estimationMethod = request.form['estimationMethod']
	startDate = request.form['startDate']
	endDate = request.form['endDate']
	

	s = pd.Series([startDate,endDate])
	s = pd.to_datetime(s,format='%Y-%m-%d')
    
	print("Start Date : ",startDate )
	print("End Date : ",endDate)
    
	estimationMethodWeight =  request.form['estimationMethodWeight']
	setWeightManually = request.form['setWeightManually']
	portfoloIndex = int(pickedStock)
	num_assets = l_data
	noWeight = False
	noManually = False
	expected_return = [] 
	weights = [] 
	annual_returns= []
	list2 =[]
	market_annual_returns = []
	risk_free_return=risk_free_return = []
	originalResult = []

	if (estimationMethodWeight=="random"):
		weights = np.random.random(num_assets)
		weights = weights / sum(weights)
	elif (estimationMethodWeight=="portfolio"):
		if(PickedStock[portfoloIndex][0]==[]):
			return render_template("noWeightInPortfolio.html")		
		else:
			weights = PickedStock[portfoloIndex][0]
			weights = np.array(weights)
			weights = weights.astype(np.float)
			weights = weights / sum(weights)
	elif (estimationMethodWeight=="manually"):
		weight = setWeightManually.split(",")  
		if(len(weight) != len(tickers)):
			return render_template("noOrWorngWeightSetting.html")			
		else:
			try:
				weights = weight
				weights = np.array(weights)
				weights = weights.astype(np.float)			
				weights = weights / sum(weights)
			except:
				return render_template("noOrWorngWeightSetting.html")			
			
    

	print(estimationMethod)

	mask = (df['Date'] >= s[0]) & (df['Date'] <=s[1])

	n_df = df.loc[mask]
		#print(n_df)
        
	if estimationMethod == "standard":
     	# simple daily returns with .pct_change() method
		cols = n_df.columns
		l = len(cols)
		daily_simple_returns = n_df.iloc[:,1:l].pct_change()  
		daily_simple_returns['Date'] = n_df['Date']
    
    	# resetting index
		daily_simple_returns.set_index(daily_simple_returns['Date'])
		daily_simple_returns = daily_simple_returns[cols]
		#print(daily_simple_returns)

		# annualise daily returns. 250 trading days in a year
		annual_returns = daily_simple_returns.mean() * 250
		print("The annualised daily returns 250 trading days in a year")
		print(annual_returns)

		# number of assets in the randomly selected portfolio
		num_assets = l_data
		print("The number of selected assets in the portfolio:")
		print(num_assets)

		# sum of weights must equal 1. 
		# (a / a+b) + (b / a+b) = 1 
		# applying this logic above

		
		print("The random weight of selected stocks:")
		print(weights)
		
		# check if the sum of weights is indeed = 1
		print("Check the sum of weights for the portfolio:")
		print(sum(weights))

		cov_matrix_d = daily_simple_returns.cov()
		cov_matrix_a = cov_matrix_d * 250

		# calculate the variance and risk of the portfolo
		port_variance = np.dot(weights.T, np.dot(cov_matrix_a, weights))
		port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_a, weights)))

		percent_var = str(round(port_variance, 4) * 100) + '%'
		percent_vols = str(round(port_volatility, 4) * 100) + '%'

		# calculate expected returns of the portfolio 
		port_returns_expected = np.sum(weights * annual_returns)
		port_returns_expected

		# convert the float into a percentage cos why not ;)
		print("The expdected returns of the portfolio (stanard):")
		print(str(round(port_returns_expected * 100, 2)) + '%')

		expected_return = str(round(port_returns_expected * 100, 2)) + '%'

		OptimizationResult.clear()
		risk=[]
		startDate_list=[]
		endDate_list=[]
		for i in range(len(tickers)):
			startDate_list.append(startDate)
			endDate_list.append(endDate)

		portfoloIndex = int(pickedStock)
		OptimizationResult.append(weights.tolist())
		OptimizationResult.append(tickers)
		OptimizationResult.append(startDate_list)
		OptimizationResult.append(endDate_list)
		OptimizationResult.append(annual_returns.tolist())	
		OptimizationResult.append(risk)
		OptimizationResult.append(portfoloIndex)

		return render_template("portfolio2.html",optimizationResult = OptimizationResult,originalResult = PickedStock[portfoloIndex],tickers = json.dumps(tickers), beta = json.dumps(list2) , market_annual_returns= market_annual_returns,risk_free_return=risk_free_return , expected_return = expected_return , weights= json.dumps(weights.tolist()) ,annual_returns= json.dumps(annual_returns.tolist()) )		

	if estimationMethod == "CAPM":
		p = pathlib.Path().absolute()
        
		p1 = str(p)
		print(p1)
		path = p1.replace('\\', '/')
		print(path)

#		print("CAPM")
		#spy=pd.read_csv(path+"/SPY.csv")
		spy = pd.DataFrame(SPY2003To2006)
		spy['Open'] = pd.to_numeric(spy['Open'],errors = 'coerce')
		spy['High'] = pd.to_numeric(spy['High'],errors = 'coerce')
		spy['Low'] = pd.to_numeric(spy['Low'],errors = 'coerce')
		spy['Close'] = pd.to_numeric(spy['Close'],errors = 'coerce')
		spy['Volume'] = pd.to_numeric(spy['Volume'],errors = 'coerce')
		spy['Adj Close'] = pd.to_numeric(spy['Adj Close'],errors = 'coerce')
		
		print (spy.dtypes)
		
		m_spy = pd.DataFrame(spy, columns = ['Date', 'Close'])
		m_spy["Date"] = pd.to_datetime(m_spy["Date"])
		mask2 = (m_spy['Date'] >= s[0]) & (m_spy['Date'] <=s[1])
		n_m_spy = m_spy.loc[mask2]
        
		risk_free_return = 0.05

     	# simple daily returns with .pct_change() method

		cols = n_df.columns
		l = len(cols)
		daily_simple_returns = n_df.iloc[:,1:l].pct_change()  
		daily_simple_returns['Date'] = n_df['Date']
        
		daily_simple_returns_market = n_m_spy.iloc[:,1].pct_change()


    	# resetting index
		daily_simple_returns.set_index(daily_simple_returns['Date'])
		daily_simple_returns = daily_simple_returns[cols]
		#print(daily_simple_returns)
        
		daily_simple_returns_market = daily_simple_returns_market.to_frame()
		daily_simple_returns_market['Date'] = n_m_spy['Date']
		daily_simple_returns_market.set_index(daily_simple_returns_market['Date'])

		# annualise daily returns. 250 trading days in a year
		# annualise daily returns. 250 trading days in a year
		annual_returns = daily_simple_returns.mean() * 250
		print("The annualised daily returns 250 trading days in a year")
		print(annual_returns)


		market_annual_returns = daily_simple_returns_market.mean() * 250
		market_annual_returns =market_annual_returns.values
		#print(market_annual_returns)

		print("Expected Annual Market Return: ",market_annual_returns)
		print("Risk free return: ",risk_free_return)


		df5 = pd.merge(daily_simple_returns, daily_simple_returns_market, on="Date")
		#print(df5)
        #Drop null values
		data = df5.dropna()
		data = data.drop(['Date'], axis=1)
		#print(data)
        
       
        
		cols_1 = df5.columns

		cols_12 = []
		for i in range(0,len(cols_1)-2):
			cols_12.append(cols_1[i])
		#print(cols_12)
        
        #Generate covarience matrix
        #print the covariance of each picked stock
		nn_df5 = data.apply(lambda cols_12: data['Close'].cov(cols_12))

		# number of assets in the randomly selected portfolio
		num_assets = l_data
		print("The number of selected assets in the portfolio:")
		print(num_assets)

         #Calc beta from list
		list1 = list(nn_df5)
#		print(list1)
		list2 = []
        
         #Calc beta from list
		for i in range(0,len(list1)-1):
			list2.append(list1[i]/list1[len(list1)-1])
		print("Beta from formula: ", list2)
		

		# sum of weights must equal 1. 
		# (a / a+b) + (b / a+b) = 1 
		# applying this logic above

		print("The random weight of selected stocks:")
		print(weights)

		# check if the sum of weights is indeed = 1
		print("Check the sum of weights for the portfolio:")
		print(sum(weights))

        
        #Calc expected return
#		expected_return = risk_free_return + beta*(data['daily_simple_returns_market'].mean()*250-risk_free_return)
		w_ = len(weights)
		cal = 0
		for i in range(w_):
			cal += weights[i]*list2[i]*(market_annual_returns-risk_free_return) 
#			print(i)
		Portfo_expected_return =risk_free_return + cal

		print("The expected returns of the portfolio (CAPM):",Portfo_expected_return)  	

		expected_return = Portfo_expected_return
		expected_return = str(round(expected_return[0] * 100, 2)) + '%'

		market_annual_returns
		market_annual_returns = str(round(market_annual_returns[0] * 100, 2)) + '%'

		OptimizationResult.clear()
		risk=[]
		startDate_list=[]
		endDate_list=[]
		for i in range(len(tickers)):
			startDate_list.append(startDate)
			endDate_list.append(endDate)
		portfoloIndex = int(pickedStock)
		OptimizationResult.append(weights.tolist())
		OptimizationResult.append(tickers)
		OptimizationResult.append(startDate_list)
		OptimizationResult.append(endDate_list)
		OptimizationResult.append(annual_returns.tolist())	
		OptimizationResult.append(risk)
		OptimizationResult.append(portfoloIndex)

		return render_template("portfolio2.html",estimationMethod = estimationMethod,optimizationResult = OptimizationResult,originalResult = PickedStock[portfoloIndex],tickers = json.dumps(tickers), beta = json.dumps(list2) , market_annual_returns= market_annual_returns,risk_free_return=risk_free_return , expected_return = expected_return , weights= json.dumps(weights.tolist()) ,annual_returns= json.dumps(annual_returns.tolist()) )				

	return redirect(url_for('home'))


@app.route('/optimization', methods = ['POST'])
def optimization():
	pickedStock = request.form.get('pickedStock2') # take portfolio data
	try:
		data = pickedStockData(int(float(pickedStock)))
	except:		
		return render_template("noPickPortfolio.html")		
	
	startDate = request.form['startDate2']
	endDate = request.form['endDate2']

	
	s = pd.Series([startDate,endDate])
	s = pd.to_datetime(s,format='%Y-%m-%d')
	
	data1=[]
	for i in range(len(data)):
		mask = (data[i]['Date'] >= s[0]) & (data[i]['Date'] <=s[1])
		n_df = data[i].loc[mask]
		print(type(n_df))		
		n_df = n_df.reset_index(drop=True)
		data1.append(n_df)

	print(data1)
	
	fixedReturn = float(request.form['fixedReturn']) / 100 #str
	riskFreeRate = float(request.form['riskFreeRate']) / 100 #str
	period = request.form['period'] # str
	short = request.form['shortSales']
	tickers = []
	
	for i in range(len(data1)):
		tickers.append(data1[i]["Ticker"][0])
	cal = Calculations(tickers, data1)
	'''
	for i in range(len(data)):
		tickers.append(data[i]["Ticker"][0])
	cal = Calculations(tickers, data)
	'''
	ret, all_tickers_returns, method = cal.getAll(period) #	can plot avg return, can also get variance~risk

	varCovarMatrix = cal.getVarCovarMatrix()
	instance = Optimization(fixedReturn, riskFreeRate, ret, varCovarMatrix)
	if short == "true":
			weights = instance.optimize()
	if short == "false":
			weights = instance.nonShortSales()
	# if period == "daily":
	# 	method = data.Date.item()
	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
	temp = np.array(all_tickers_returns)
	df_returns = pd.DataFrame(temp.T, columns=tickers)
	df_returns.corr()
	'''
	pd.plotting.scatter_matrix(df_returns, figsize=(6, 6))
	plt.savefig('./corr.png')
	'''
	
	
	y = []
	for i in range(len(tickers)):
		y.append(cal.getVariance(all_tickers_returns[i]))
	
	''''
	fig = plt.figure()
	plt.bar(tickers, y, color=colors)
	fig.suptitle('Risk among different tickers')
	plt.xlabel('Tickers')
	plt.ylabel('Variance/Risk')
	plt.savefig('./risk.png')

	fig = plt.figure()
	plt.bar(tickers, weights, color=colors)
	fig.suptitle('Weights distrubution')
	plt.xlabel('Tickers')
	plt.ylabel('Weights')
	plt.savefig('./weights.png')
	'''

	abc4 = []
	for i in range(len(tickers)):
		abc3 = []
		for d in range(len(tickers)):
			abc2 = []
			for f in range(len(df_returns[tickers[0]])):        
				if(i!=d):                   
					abc2.append({"x":df_returns[tickers[i]][f],"y":df_returns[tickers[d]][f]})
				else:
					abc2.append(df_returns[tickers[i]][f])
			abc3.append(abc2)
		abc4.append(abc3)

	OptimizationResult.clear()
	risk = y
	
	startDate_list=[]
	endDate_list=[]
	stock_return = ""
	
	for i in range(len(tickers)):
			startDate_list.append(startDate)
			endDate_list.append(endDate)

	

	portfoloIndex = int(pickedStock)
	OptimizationResult.append(weights.tolist())
	OptimizationResult.append(tickers)
	OptimizationResult.append(startDate_list)
	OptimizationResult.append(endDate_list)
	OptimizationResult.append(stock_return)	
	OptimizationResult.append(risk)
	OptimizationResult.append(portfoloIndex)
	'''
	stock_return = ""
	startDate = []
	endDate = []

	portfoloIndex = int(pickedStock)
	OptimizationResult.append(weights.tolist())
	OptimizationResult.append(tickers)
	OptimizationResult.append(startDate)
	OptimizationResult.append(endDate)
	OptimizationResult.append(stock_return)	
	OptimizationResult.append(risk)
	OptimizationResult.append(portfoloIndex)
	'''
	print("-------------")
	print(OptimizationResult)
	print("-------------")

	#return redirect(url_for('home'))
	return render_template("portfolio.html",fixedReturn= fixedReturn,riskFreeRate=riskFreeRate, tickersLen = len(tickers), tickers = json.dumps(tickers) , weights = json.dumps(weights.tolist()), y  = json.dumps(y), df_returns = json.dumps(abc4),optimizationResult = OptimizationResult,originalResult = PickedStock[portfoloIndex] )		

class graph: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y

if __name__ == '__main__':
	app.run(port=5000, debug = True)

