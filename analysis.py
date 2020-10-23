import json
import datetime
import statistics

from collections import Counter
import operator
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from model import Model

class JsonData:

    # Constructor takes in the filename of a JSON to be parsed
    def __init__(self, filename, ignored = ('Credit'), model = None):
        with open(filename) as json_file:
            data = json.load(json_file)
        self.category_dict = {}
        self.date_dict = {}
        self.deposit_cats = {'Deposit', 'ACH Deposit'}
        self.model = model

        for transaction in data['transactions']:

            if len(transaction['category']) == 1:
                cat_idx = 0
            else:
                cat_idx = 1

            category = transaction['category'][cat_idx]
            if category in ignored:
                continue

            amount = transaction['amount']
            date = transaction['date']


            if category in self.category_dict:
                self.category_dict[category].append({'amount': amount, 'date': date})
            else:
                self.category_dict[category] = [{'amount': amount, 'date': date}]

            date = self.date_to_datetime(date)

            if date in self.date_dict:
                self.date_dict[date].append({'amount': amount, 'category': category})
            else:
                self.date_dict[date] = [{'amount': amount, 'category': category}]


        for key in self.category_dict:
            self.category_dict[key].sort(key=lambda x: x['date'])

        for key in self.date_dict:
            self.date_dict[key].sort(key=lambda x: x['category'])


    # Prints all the categories and their expenditures, with amount and date
    def __str__(self):
        return_string = ''
        for key, val in self.category_dict.items():
            return_string += '{}: {}\n'.format(key, val)
        return return_string

    # Takes in a date in string format aligned with the json and returns a datetime object. If the date is already a
    # datetime object, returns that same object.
    # Precondition: date is well formatted, i.e.
    # is a string of the form 'xxxx-yy-zz' where xxxx is the year, yy is the month, zz is the day
    def date_to_datetime(self, date):
        if type(date) == type(datetime.date.today()):
            return date
        y, m, d = date.split('-')
        return datetime.date(int(y), int(m), int(d))

    # Returns a sorted list of all the categories in category_dict
    def list_categories(self):
        return sorted(list(self.category_dict.keys()))

    # Returns category_dict[category]
    def __getitem__(self, category):
        return self.category_dict[category]

    # Returns a dictionary whose keys are categories and whose values are the total expenditures for that category.
    # Takes in optional argument 'category' which, if supplied, only returns the sum of expenditures for that category.
    def sum_expenditures(self, category=None):

        if category is not None:
            return sum(x['amount'] for x in self.category_dict[category])

        expenditures = {}
        for key, val in self.category_dict.items():
            expenditures[key] = round(sum(x['amount'] for x in val), 2)

        return expenditures

    # Returns the category with the largest expense
    def largest_expense(self):

        expenditures = self.sum_expenditures()
        return max(expenditures, key=expenditures.get)

    # Returns the category with the smallest expense
    def smallest_expense(self):

        expenditures = self.sum_expenditures()
        return min(expenditures, key=expenditures.get)

    # Returns the number of expenditures by category between the two dates
    # Precondition: dates are well-formated as they are in date_to_datetime
    def num_between_dates(self, start_date, end_date):
        start = self.date_to_datetime(start_date)
        end = self.date_to_datetime(end_date)

        result = dict([(x, 0) for x in self.list_categories()])

        for time, transactions in self.date_dict.items():
            if time >= start and time <= end:
                for transaction in transactions:
                    result[transaction['category']] += 1

        return result

    # Returns the sum of expenditures by category between the two dates. Takes in optional argument
    # 'category' that, if provided, returns the sum of expenditures between dates for only that category.
    # Precondition: dates are well-formated as they are in date_to_datetime
    def sum_between_dates(self, start_date, end_date, category=None):
        start = self.date_to_datetime(start_date)
        end = self.date_to_datetime(end_date)

        if category is not None:
            result = 0
            for transaction in self.category_dict[category]:
                date = self.date_to_datetime(transaction['date'])
                if date >= start and date <= end:
                    result += transaction['amount']
            return round(result, 2)

        result = dict([(x, 0) for x in self.list_categories()])

        for time, transactions in self.date_dict.items():
            if time >= start and time <= end:
                for transaction in transactions:
                    result[transaction['category']] += transaction['amount']

        for expenditure, value in result.items():
            if value < 0:
                result[expenditure] = 0
            else:
                result[expenditure] = round(value, 2)

        return result

    # Returns the average (by day) expenditure between the two dates
    # Precondition: dates are well-formated as they are in date_to_datetime
    def avg_between_dates(self, start_date, end_date, category=None):
        if category is not None:
            total = self.sum_between_dates(start_date, end_date, category)
            return round(total / ((self.date_to_datetime(end_date) - self.date_to_datetime(start_date)).days + 1), 2)

        sum_dict = self.sum_between_dates(start_date, end_date)
        return round(sum(sum_dict.values()) /
                     ((self.date_to_datetime(end_date) - self.date_to_datetime(start_date)).days + 1), 2)

    # Returns the deposit frequency (by day) between two dates. Optionally modify the 'deposit_cats' argument
    # if you would like more categories to count as deposits.
    # Precondition: dates are well-formated as they are in date_to_datetime
    def deposit_freq(self, start_date, end_date):
        start = self.date_to_datetime(start_date)
        end = self.date_to_datetime(end_date)

        total = 0

        for cat, transactions in self.category_dict.items():
            if cat in self.deposit_cats:
                total += len(transactions)

        return round(total / ((end - start).days + 1), 2)

    # Returns the average deposit (by day) between two dates. Optionally modify the 'deposit_cats' argument
    # if you would like more categories to count as deposits.
    # Precondition: dates are well-formated as they are in date_to_datetime
    def avg_deposit(self, start_date, end_date):
        start = self.date_to_datetime(start_date)
        end = self.date_to_datetime(end_date)

        total = 0

        for cat, transactions in self.category_dict.items():
            if cat in self.deposit_cats:
                for transaction in transactions:
                    total += transaction['amount']

        return round(total / ((end - start).days + 1), 2)

    # Returns the average deposit and the by-day frequency of deposits using a (months)-day lookback
    # Precondition: 1 <= months <= 12
    def lookback(self, months):
        end = datetime.date.today()
        if end.month <= months:
            start = datetime.date(end.year-1, 13-months, end.day)
        else:
            start = datetime.date(end.year, end.month - months, end.day)

        avg = self.avg_between_dates(start, end)
        num = round(sum(self.num_between_dates(start, end).values()) / ((end - start).days + 1), 2)

        return avg, num

    # Returns the magnitude volatility (standard deviation) of all expenditures in a dictionary whose key is
    # the category and whose value is the volatility of expenditures in that category. If the optional argument
    # 'category' is given, only returns the volatility of that category.
    # Requirement: 'category' has at least 2 expenditures.
    def magnitude_volatility(self, start_date, end_date, category=None):

        start, end = self.date_to_datetime(start_date), self.date_to_datetime(end_date)

        if category is not None:
            values = []
            for expenditure in self.category_dict[category]:
                date = self.date_to_datetime(expenditure['date'])
                if date >= start and date <= end:
                    values.append(expenditure['amount'])
            volatility = round(statistics.stdev(values), 2)
        else:
            volatility = {}
            for category, expenditure_list in self.category_dict.items():
                values = []
                for expenditure in expenditure_list:
                    date = self.date_to_datetime(expenditure['date'])
                    if date >= start and date <= end:
                        values.append(expenditure['amount'])
                if sum(values) < 0:
                    volatility[category] = 'Negative sum of expenditures'
                elif len(values) < 2:
                    volatility[category] = 'Not enough data points'
                else:
                    volatility[category] = round(statistics.stdev(values), 2)

        return volatility


    # Returns the timing volatility (standard deviation) of all expenditures in a dictionary whose key is
    # the category and whose value is the volatility of timings in that category. If the optional argument
    # 'category' is given, only returns the volatility of that category.
    # Requirement: 'category' has at least 3 expenditures.
    def timing_volatility(self, start_date, end_date, category=None):

        start, end = self.date_to_datetime(start_date), self.date_to_datetime(end_date)

        if category is not None:
            values = []
            for expenditure in self.category_dict[category]:
                date = self.date_to_datetime(expenditure['date'])
                if date >= start and date <= end:
                    values.append(self.date_to_datetime(expenditure['date']))
            values.sort()
            timings = [(end-start).days for start, end in zip(values[:-1], values[1:])]
            volatility = round(statistics.stdev(timings))
        else:
            volatility = {}
            for category, expenditure_list in self.category_dict.items():
                values = []
                for expenditure in expenditure_list:
                    date = self.date_to_datetime(expenditure['date'])
                    if date >= start and date <= end:
                        values.append(self.date_to_datetime(expenditure['date']))
                values.sort()
                timings = [(end-start).days for start, end in zip(values[:-1], values[1:])]
                if len(timings) < 2:
                    volatility[category] = 'Not enough data points'
                else:
                    volatility[category] = round(statistics.stdev(timings))

        return volatility

    # Returns a dictionary whose keys are categories and whose values are the percentages of the total expenditures belonging
    # to that respective category.
    def percentage_expenditures(self):
        expenditures = self.sum_expenditures()

        for expenditure, value in expenditures.items():
            if value < 0:
                del expenditures[expenditure]

        total = sum(expenditures.values())

        return dict([(k, round(v/total * 100, 2)) for (k, v) in expenditures.items()])

    # Returns the magnitude and timing volatilities for the 'Debit' category.
    def paycheck_consistency(self, start_date, end_date):

        try:
            mag = self.magnitude_volatility(start_date, end_date, 'Debit')
            tim = self.timing_volatility(start_date, end_date, 'Debit')
        except KeyError:
            return None, None

        return mag, tim

    # Returns the free cash flow (FCF) between start_date and end_date
    def free_cash_flow(self, start_date, end_date):
        start = self.date_to_datetime(start_date)
        end = self.date_to_datetime(end_date)

        sums = self.sum_between_dates(start_date, end_date)

        try:
            deposit = sums.pop('Deposit')
        except KeyError:
            deposit = 0

        try:
            debit = sums.pop('Debit')
        except KeyError:
            debit = 0

        expenses = sum(sums.values())

        return round((deposit + debit) / (expenses + 1) , 2)

    # Returns a tuple containing the free cash flow in [period]-long increments as a list, and the standard deviation of
    # that list.
    # Requires: period is 'biweekly' or 'monthly'
    def free_cash_flow_period(self, start_date, end_date, period):

        assert period in ['biweekly', 'monthly']

        start = self.date_to_datetime(start_date)
        end = self.date_to_datetime(end_date)

        current_start = start
        current_end = start

        fcf = []

        while current_end < end:

            current_start = current_end

            if period == 'biweekly':
                current_end = current_end + datetime.timedelta(days=14)
            elif period == 'monthly':
                if current_end.month == 12:
                    current_end = datetime.date(current_end.year + 1, 1, current_end.day)
                else:
                    current_end = datetime.date(current_end.year, current_end.month + 1, current_end.day)

            if current_end > end:
                current_end = end

            fcf.append(self.free_cash_flow(current_start, current_end))

        return fcf, round(statistics.stdev(fcf), 2)


    # Returns the credit card to income (CCI) between start_date and end_date
    def credit_card_to_income(self, start_date, end_date):
        start = self.date_to_datetime(start_date)
        end = self.date_to_datetime(end_date)

        deposit = self.sum_between_dates(start_date, end_date, 'Deposit')
        debit = self.sum_between_dates(start_date, end_date, 'Debit')
        credit = self.sum_between_dates(start_date, end_date, 'Credit Card')

        return round((deposit + debit) / credit, 2)


    def expenditures_between_dates(self, start_date, end_date):
        start = self.date_to_datetime(start_date)
        end = self.date_to_datetime(end_date)

        result = {k:[] for k in self.list_categories()}

        for date, expenditures in self.date_dict.items():
            if date >= start and date <= end:
                for expenditure in expenditures:
                    result[expenditure['category']].append(expenditure['amount'])

        return result

    # Plots a bar graph containing the total per-category expenses between a date range.
    # Additionally overlays the relative frequencies of expenditures.
    def plot_expenses_between_dates(self, start_date, end_date):

        expense_dict = self.sum_between_dates(start_date, end_date)
        freq_dict = self.num_between_dates(start_date, end_date)

        scale = max(expense_dict.values()) / max(freq_dict.values())

        plt.figure(figsize=(19, 5))
        plt.bar(expense_dict.keys(), expense_dict.values())
        plt.scatter(range(len(expense_dict)), [x * scale for x in freq_dict.values()],
                    c='red', zorder=2, marker='s', edgecolors='black', s=100)
        plt.show()

    # Prints a table containing the average, minimum, maximum, frequency, magnitude volatility, and timing volatility
    # for all categories in a given date range.
    def table_data_between_dates(self, start_date, end_date):

        expense_dict = self.sum_between_dates(start_date, end_date)
        freq_dict = self.num_between_dates(start_date, end_date)

        avg_dict = {k:round(expense_dict[k] / freq_dict[k], 2) for k in expense_dict.keys()}
        mag_dict = self.magnitude_volatility(start_date, end_date)
        tim_dict = self.timing_volatility(start_date, end_date)

        expenditure_dict = self.expenditures_between_dates(start_date, end_date)

        min_dict = {k:min(expenditure_dict[k]) for k in expenditure_dict.keys()}
        max_dict = {k:max(expenditure_dict[k]) for k in expenditure_dict.keys()}

        table = [['', 'Average', 'Minimum', 'Maximum', 'Frequency', '$ Vol', 'Timing Vol']]

        for cat in self.category_dict.keys():
            table.append([cat, avg_dict[cat], min_dict[cat], max_dict[cat], freq_dict[cat], mag_dict[cat], tim_dict[cat]])

        print(tabulate(table))

    # Plots a pie chart outlining the expenditures in a given date range
    def pie_chart(self, start_date, end_date):

        expense_dict = self.sum_between_dates(start_date, end_date)

        plt.pie(list(expense_dict.values()), labels=expense_dict.keys(), autopct='%1.1f%%')
        plt.show()

    # Plots the expenses over time for a given category over a given date range
    def plot_expenses_over_time(self, category, start_date, end_date):

        start = self.date_to_datetime(start_date)
        end = self.date_to_datetime(end_date)

        good_transactions = {}

        for transaction in self.category_dict[category]:
            date = self.date_to_datetime(transaction['date'])
            if date >= start and date <= end:
                days = (date - start).days
                if days not in good_transactions:
                    good_transactions[days] = transaction['amount']
                else:
                    good_transactions[days] += transaction['amount']

        days_value_list =  list(sorted(good_transactions.items()))

        x, y = np.array([t[0] for t in days_value_list]), np.array([t[1] for t in days_value_list])
        b, m = np.polynomial.polynomial.polyfit(x, y, 1)

        plt.plot(x, y, '.')
        plt.plot(x, b + m * x, '-')

        plt.title(category + ' Expenses (' + start_date + ' to ' + end_date + ')')
        plt.ylabel('Expenses')
        plt.xlabel('Days since ' + start_date)

        plt.show()


    # Plots the weekly net income analysis over a given date range
    def weekly_net_income_analysis(self, start_date, end_date):

        start = self.date_to_datetime(start_date)
        end = self.date_to_datetime(end_date)

        current_start = start
        current_end = start

        net_income = []

        income_cats = {'Debit'}

        while current_end < end:

            current_start = current_end
            current_end = current_end + datetime.timedelta(days=7)

            if current_end > end:
                current_end = end

            sums = self.sum_between_dates(current_start, current_end)

            debit = sums.pop('Debit')
            deposit = sums.pop('Deposit')

            net_income.append(debit + deposit - sum(sums.values()))

        plt.plot(range(len(net_income)), net_income)
        plt.title('Weekly Net Income analysis (' + start_date + ' to ' + end_date + ')')
        plt.ylabel('Net Income')
        plt.xlabel('Weeks since ' + start_date)
        plt.show()


    # Returns the periodic sum of expenditures by category with a period-day increment
    # Subfunction for monte carlo
    def periodic_sum(self, period=14):

        current_start = min(test.date_dict.keys())
        current_end = current_start - datetime.timedelta(days=1)
        end = max(test.date_dict.keys())

        net_income = []

        result = {}

        while current_end < end:

            current_start = current_end + datetime.timedelta(days=1)
            current_end = current_end + datetime.timedelta(days=period)

            if current_end > end:
                current_end = end

            sums = self.sum_between_dates(current_start, current_end)

            for k, v in sums.items():
                if k not in result:
                    result[k] = []
                result[k].append(v)

        return result

    # Performs a monte carlo simulation of an account balance
    # Returns a matrix where rows represent simulations and columns represent tenors, the min, max, and average account
    #     balances across all simulations, and the probability of defaulting
    def monte_carlo(self, n_sims = 10000, loan_amount = 1, loan_rate = None):

        assert self.model is not None

        temp_loan_amount = model.loan_amount
        temp_full_rate_2 = model.full_rate_2

        model.loan_amount = loan_amount * temp_loan_amount
        if loan_rate is not None:
            model.temp_full_rate_2 = loan_rate

        init_bal = self.model.current_account_balance
        loan_tenor = self.model.loan_tenor
        cat_dict = self.periodic_sum()

#         print('init bal: {}'.format(init_bal))
#         print('loan tenor: {}'.format(loan_tenor))
#         print('base data: {}'.format(cat_dict))

        result = np.full((n_sims, loan_tenor), init_bal) - np.cumsum(self.model.base_loan_cf.info_array[5, 1:])

#         print('base result: \n{}'.format(result))

        # What this does is figure out the mean and std of the input vector and generate a (n_sims x tenor) matrix
        # drawn from this sample. It then takes the maximum of that vector and 0, essentially turning all negative values
        # into 0. It then takes a cumulative sum across the columns, as that will represent a rolling account balance
        for k, v in cat_dict.items():
            val = np.cumsum(np.maximum(np.random.normal(np.mean(v), np.std(v), (n_sims, loan_tenor)), 0), axis=1)
            if k in self.deposit_cats:
                result = np.add(result, val)
            else:
                result = np.subtract(result, val)

        mn = np.min(result, axis=0)
        mx = np.max(result, axis=0)
        av = np.mean(result, axis=0)
        pr = np.sum(np.min(result, axis=1) < 0) / n_sims

        model.full_rate_2 = temp_full_rate_2
        model.loan_amount = temp_loan_amount

        return result, mn, mx, av, pr

    def monte_carlo_matrix(self, n_sims=10000):

        result_matrix = np.zeros((8, 7))

        # Way to loop from 92.50% to 110.00%
        for i, loan_amount in enumerate(range(185, 225, 5)):
            loan_amount /= 200
            # Way to loop from
            for j, loan_rate in enumerate(np.arange(7, 14)):
                loan_rate /= 200
                _, _, _, _, res = self.monte_carlo(n_sims=n_sims, loan_amount = loan_amount, loan_rate = loan_rate)
                result_matrix[i][j] = res

        return result_matrix


if __name__ == '__main__':
    # model = Model(loan_amount=1000, loan_tenor=4, promo_rate=0.0, full_rate_2=.24, promo_period=0,
    #           amortization_type=1, current_account_balance=2000, avg_deposit_3mth=1200,
    #           avg_weekly_expense_3mth=500, deposit_frequency=2, deposit_only_stress=0.8,
    #           expense_only_stress=1.30, two_leg_deposit_stress=1.0, two_leg_expense_stress=1.25,
    #           timed_deposit_shock=0.75, timed_deposit_shock_weeks=2, timed_expense_shock=1.25,
    #           timed_expense_shock_1_weeks=1, timed_expense_shock_2_weeks=3)
    #
    # test = JsonData("generated_data.json", model=model)
    # # print('All data:')
    # # print(test)
    #
    # print('\nList of categories:')
    # print(test.list_categories())
    #
    # # print('\nGetting data for a specific category:')
    # # print(test['Car Service'])
    #
    # print('\nSum of expenditures for each category:')
    # print(test.sum_expenditures())
    #
    # print('\nLargest expense:')
    # print(test.largest_expense())
    #
    # print('\nSmallest expense:')
    # print(test.smallest_expense())
    #
    # print('\nNumber of expenditures between two dates:')
    # print(test.num_between_dates('2019-10-14', '2019-10-16'))
    #
    # print('\nSum of expenditures between two dates:')
    # print(test.sum_between_dates('2019-10-14', '2019-10-16'))
    #
    # print('\nSum of expenditures between two dates for a given category:')
    # print(test.sum_between_dates('2019-10-14', '2019-10-16', 'Restaurants'))
    #
    # print('\nAverage expenditure between two dates:')
    # print(test.avg_between_dates('2019-10-14', '2019-10-16'))
    #
    # print('\nAverage expenditure between two dates for a given category:')
    # print(test.avg_between_dates('2019-10-14', '2019-10-16', 'Restaurants'))
    #
    # print('\nDeposit frequency between two dates:')
    # print(test.deposit_freq('2019-10-14', '2019-10-16'))
    #
    # print('\nAverage deposit between two dates:')
    # print(test.avg_deposit('2019-10-14', '2019-10-16'))
    #
    # print('\n3-month lookback')
    # print(test.lookback(3))
    #
    # print('\nMagnitude volatility between two dates:')
    # print(test.magnitude_volatility('2018-10-14', '2019-10-16'))
    #
    # print('\nMagnitude volatility for a specific category between two dates:')
    # print(test.magnitude_volatility('2018-10-14', '2019-10-16', 'Restaurants'))
    #
    # print('\nTiming volatility betweeen two dates:')
    # print(test.timing_volatility('2018-10-14', '2019-10-16'))
    #
    # print('\nTiming volatility for a specific category between two dates:')
    # print(test.timing_volatility('2018-10-14', '2019-10-16', 'Restaurants'))
    #
    # print('\nPercentage Expenditures:')
    # print(test.percentage_expenditures())
    #
    # print('\nPaycheck Consistency:')
    # print(test.paycheck_consistency('2018-10-14', '2019-10-16'))
    #
    # print('\nFree Cash Flow:')
    # print(test.free_cash_flow('2019-10-14', '2019-11-30'))
    #
    # print('\nBiweekly Free Cash Flow:')
    # print(test.free_cash_flow_period('2019-10-14', '2019-11-30', 'biweekly'))
    #
    # print('\nMonthly Free Cash Flow:')
    # print(test.free_cash_flow_period('2018-10-14', '2019-11-30', 'monthly'))
    #
    # print('\nCredit Card to Income:')
    # print(test.credit_card_to_income('2019-10-14', '2019-11-30'))
    #
    # print('\nGetting all expenditures by category in a given date range:')
    # print(test.expenditures_between_dates('2019-10-14', '2019-11-30'))
    #
    # print('\nPlotting expenses between dates:')
    # test.plot_expenses_between_dates('2019-10-14', '2019-11-30')
    #
    # print('\nPlotting table data between dates:')
    # test.table_data_between_dates('2019-10-14', '2019-11-30')
    #
    # print('\nPie chart between dates:')
    # test.pie_chart('2019-10-14', '2019-11-30')
    #
    # print('\nExpenses for Restaurants between dates:')
    # test.plot_expenses_over_time('Restaurants', '2019-10-14', '2019-11-30')
    #
    # print('\nWeekly net income analysis between dates:')
    # test.weekly_net_income_analysis('2018-10-14', '2019-11-30')
    #
    # print('\nMonte carlo approximation:')
    # res, mn, mx, av, p = test.monte_carlo()
    # print('Resulting matrix: \n{}'.format(res))
    # print('Min account balance: {}'.format(mn))
    # print('Max account balance: {}'.format(mx))
    # print('Avg account balance: {}'.format(av))
    # print('Probability of defaulting: {}'.format(p))

    model = Model(loan_amount=9000, loan_tenor=4, promo_rate=0.0, full_rate_2=.24, promo_period=0,
              amortization_type=2, current_account_balance=2000, avg_deposit_3mth=1200,
              avg_weekly_expense_3mth=500, deposit_frequency=2, deposit_only_stress=0.8,
              expense_only_stress=1.30, two_leg_deposit_stress=1.0, two_leg_expense_stress=1.25,
              timed_deposit_shock=0.75, timed_deposit_shock_weeks=2, timed_expense_shock=1.25,
              timed_expense_shock_1_weeks=1, timed_expense_shock_2_weeks=3)

test = JsonData("simple_data.json", model=model)

print('\nList of categories:')
print(test.list_categories())

print('\nSum of expenditures for each category:')
print(test.sum_expenditures())

print('\nLargest expense:')
print(test.largest_expense())

print('\nSmallest expense:')
print(test.smallest_expense())

print('\nNumber of expenditures between two dates:')
print(test.num_between_dates('2020-03-01', '2020-03-31'))

print('\nSum of expenditures between two dates:')
print(test.sum_between_dates('2020-03-01', '2020-03-31'))

print('\nTiming volatility betweeen two dates:')
print(test.timing_volatility('2020-03-01', '2020-03-31'))

print('\nTiming volatility for a specific category between two dates:')
print(test.timing_volatility('2020-03-01', '2020-03-31', 'Restaurants'))

print('\nPaycheck Consistency:')
print(test.paycheck_consistency('2020-03-01', '2020-03-31'))

print('\nGetting all expenditures by category in a given date range:')
print(test.expenditures_between_dates('2020-03-01', '2020-03-31'))

print('\nPlotting expenses between dates:')
test.plot_expenses_between_dates('2020-03-01', '2020-03-31')

print('\nPlotting table data between dates:')
test.table_data_between_dates('2020-03-01', '2020-03-31')

print('\nPie chart between dates:')
test.pie_chart('2020-03-01', '2020-03-31')

print('\nExpenses for Restaurants between dates:')
test.plot_expenses_over_time('Restaurants', '2020-03-01', '2020-03-31')

print('\nMonte carlo approximation:')
res, mn, mx, av, p = test.monte_carlo()
print('Resulting matrix: \n{}'.format(res))
print('Min account balance: {}'.format(mn))
print('Max account balance: {}'.format(mx))
print('Avg account balance: {}'.format(av))
print('Probability of defaulting: {}'.format(p))

print('Monte Carlo matrix: \n{}'.format(test.monte_carlo_matrix()))
