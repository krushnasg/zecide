from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField
from wtforms.validators import DataRequired

indicators = [ 'EMA',
            'DEMA',
            'WILLR',
            'ADX',
            'MACD',
            'BBANDS']
indicators.sort()

indices = [ 'TCS',
            'ABCAPITAL',
            'RELI',
            'SPARC']


class IndicatorForm(FlaskForm):
    indicatorList = [(idx,idx) for idx in indicators]
    indexList = [(idx,idx) for idx in indices]
    mIndex = SelectField('Index', choices = indexList, default = 'TCS', validators=[DataRequired()])
    indicatorType  = SelectField('Indicator', choices = indicatorList, default = 'EMA', validators=[DataRequired()])

    # username = StringField('Username', validators=[DataRequired()])
    # password = PasswordField('Password', validators=[DataRequired()])
    # remember_me = BooleanField('Remember Me')
    submit = SubmitField('GO')