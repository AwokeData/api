import time
from flask import Flask
import analysis

app = Flask(__name__)


@app.route('/get_data')
def get_json_data():
	res = {}
	test = analysis.JsonData("generated_data.json")
	res["categories"] = test.list_categories()
	res["largest_expenditure"] = test.largest_expense()
	res["smallest_expenditure"] = test.smallest_expense()
	res["sum_expenditures"] = test.sum_expenditures()
	res["percentage_expenditures"] = test.percentage_expenditures()
	res["lookback"] = test.lookback(3)

	return res
