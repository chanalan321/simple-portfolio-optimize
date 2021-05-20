import numpy as np
import pandas as pd
from datetime import datetime

class Calculations():
	def __init__(self, tickers, data):
		super(Calculations, self).__init__()
		self.tickers = tickers
		self.data = data
		self.allReturns = []

	def getAll(self, s):
		avg_return = [] # list of avg returns
		for each_df in self.data:
			returns = []
			yearList = self.getNumberOfYear(each_df.Date)
			monthList = []
			for i in yearList:
				monthList.append(self.getNumberOfMonth(i, each_df.Date))
			if s == "monthly":
				final = []
				for dex in range(len(yearList)):
					for m in monthList[dex]:
						final.append(self.getFirstLastDayOfMonth(each_df.Date, yearList[dex], m))
				for k in final:
					returns.append(self.getReturnForEachMonth(each_df, k))
				avg_return.append(sum(returns) / len(final))
			if s == "yearly":
				for yearly in yearList:
					returns.append(self.getReturnForEachYear(yearly, each_df))
				avg_return.append(sum(returns) / len(yearList))
			if s == "daily":
				dif = each_df.Close.diff().tolist()
				dif = dif[1:]
				start = each_df.Close.tolist()
				start = start[:-1]
				each_returns = np.divide(dif, start)
				returns = list(each_returns)
				avg_return.append(sum(returns) / each_df.Date.size)
			# variance.append(self.getVariance(returns))
			self.allReturns.append(returns)
		# self.getCorrelation()
		return avg_return, self.allReturns, yearList

	def getReturnForEachYear(self, y, data):
		temp = []
		result = []
		for i in data.Date:
			if i.year == y:
				temp.append(i)
		start_date = temp[0]
		end_date = temp[len(temp) - 1]
		start = data.loc[data["Date"] == start_date].Close.item()
		end = data.loc[data["Date"] == end_date].Close.item()
		return (end - start) / start

	def getReturnForEachMonth(self, data, two_days):
		start = data.loc[data["Date"] == two_days[0]].Close.item()
		end = data.loc[data["Date"] == two_days[1]].Close.item()
		return (end - start) / start

	def getNumberOfYear(self, dates):
		yearList = []
		for i in dates:
			if i.year not in yearList:
				yearList.append(i.year)
		return yearList

	def getNumberOfMonth(self, y, dates):
		monthList = []
		for i in dates:
			if i.year == y and i.month not in monthList:
				monthList.append(i.month)
		return monthList

	def getFirstLastDayOfMonth(self, date, y, m):
		temp = []
		result = []
		for i in date:
			if i.year == y and i.month == m:
				temp.append(i)
		result.append(temp[0])
		result.append(temp[len(temp) - 1])
		return result

	def getVariance(self, lst):
		arr = np.array(lst)
		return np.var(arr)

	def getCorrelation(self, lst):
		df = pd.DataFrame(lst)
		print(df.corr())
		return

	def getVarCovarMatrix(self):
		size = len(self.tickers)
		m = []
		for i in range(size):
			row = []
			for j in range(size):
				if i == j:
					row.append(self.getVariance(self.allReturns[j]))
					continue;
				x = pd.to_numeric(pd.DataFrame(list(self.allReturns[i]))[0], downcast = "float")
				y = pd.to_numeric(pd.DataFrame(list(self.allReturns[j]))[0], downcast = "float")
				row.append(self.getVariance(self.allReturns[i]) * self.getVariance(self.allReturns[j]) * x.corr(y))
			m.append(row)
		return np.array(m)

	def getCovariance(self):
		return

if __name__ == '__main__':
	data = []
	path = "C:/Users/chanm/Desktop/comp4146_project/data/WMT.txt"
	read_file = pd.read_csv(path, parse_dates=["Date"])
	data.append(read_file)
	data.append(pd.read_csv("C:/Users/chanm/Desktop/comp4146_project/data/MMM.txt", parse_dates=["Date"]))
	name = ['WMT', 'WWW']
	cal = Calculations(name, data)
	ret = cal.getAll("daily")

	a = cal.getVarCovarMatrix()
	print(a)
