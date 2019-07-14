import os
import functions.strength_ti as st
import pandas as pd
from bs4 import BeautifulSoup
from flask import Flask,render_template, redirect, url_for,request, Markup
from forms import IndicatorForm

app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])

@app.route('/', methods = ['GET','POST'])
def hello():
    form  = IndicatorForm()
    if request.method == 'POST' and form.validate_on_submit():
            indexName = request.form.get('mIndex')
            indicatorName = request.form.get('indicatorType')
            print (indicatorName)
            data = pd.read_csv('data/' + indexName + '.NS.csv',usecols=['Date','Open', 'High', 'Low', 'Close'])
            score,fig = st.get_ti_strength(indicatorName,data)
            # soup = BeautifulSoup(fig)
            # fig = soup.prettify()
            # fig = Markup(fig)
            # fig = '<p class="title"><b>The Dormouse\'s story</b></p>'
            # with open(fig, 'r') as myfile:
            #     fig = myfile.read()
            return render_template("plot.html", form = form, score = score, fig = fig)
            
    print(os.environ['APP_SETTINGS'])
    return render_template("index.html",form = form)

@app.route('/di', methods = ['GET'])
def displayIndicator(score,fig):

    list_traces=[]
	# list_traces.append(go.Scatter(x=data['Date'], y= op[0], name = 'macd'))
	# list_traces.append(go.Scatter(x=data['Date'], y= op[1], name = 'signal line'))
	# list_traces.append(go.Bar(x=data['Date'], y= op[2], name = 'momentum histogram'))
	# fig = go.Figure(data=list_traces)
    return render_template("plot.html", )

if __name__ == '__main__':
    app.run()
