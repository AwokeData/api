
class Model:

    def __init__(self, loan_amount, loan_tenor, full_rate_2, amortization_type, promo_rate, promo_period,
                 current_account_balance, avg_deposit_3mth, avg_weekly_expense_3mth, deposit_frequency,
                 deposit_only_stress, expense_only_stress, two_leg_deposit_stress, two_leg_expense_stress,
                 timed_deposit_shock, timed_deposit_shock_weeks, timed_expense_shock, timed_expense_shock_1_weeks,
                 timed_expense_shock_2_weeks):

        from base_loan_cf import base_loan_cf
        from account_balance_base_case import account_balance_base_case
        from stress_case_1_deposit_expense import stress_case_1_deposit_expense
        from stress_case_2_deposit import stress_case_2_deposit
        from stress_case_3_expense import stress_case_3_expense
        from stress_case_4_timed_deposit import stress_case_4_timed_deposit
        from stress_case_5_timed_expense import stress_case_5_timed_expense

        # Base Loan CF
        self.loan_amount = loan_amount
        self.loan_tenor = loan_tenor
        self.full_rate_2 = full_rate_2
        self.amortization_type = amortization_type
        self.promo_rate = promo_rate
        self.promo_period = promo_period

        # Account Balance
        self.current_account_balance = current_account_balance
        self.avg_deposit_3mth = avg_deposit_3mth
        self.avg_weekly_expense_3mth = avg_weekly_expense_3mth
        self.deposit_frequency = deposit_frequency

        # Stress
        self.deposit_only_stress = deposit_only_stress
        self.expense_only_stress = expense_only_stress
        self.two_leg_deposit_stress = two_leg_deposit_stress
        self.two_leg_expense_stress = two_leg_expense_stress
        self.timed_deposit_shock = timed_deposit_shock
        self.timed_deposit_shock_weeks = timed_deposit_shock_weeks
        self.timed_expense_shock = timed_expense_shock
        self.timed_expense_shock_1_weeks = timed_expense_shock_1_weeks
        self.timed_expense_shock_2_weeks = timed_expense_shock_2_weeks

        # Running models
        self.base_loan_cf = ModelData(*base_loan_cf(self))

        self.account_balance_base_case = ModelData(*account_balance_base_case(self))

        self.stress_case_1_deposit_expense = ModelData(*stress_case_1_deposit_expense(self))

        self.stress_case_2_deposit = ModelData(*stress_case_2_deposit(self))

        self.stress_case_3_expense = ModelData(*stress_case_3_expense(self))

        self.stress_case_4_timed_deposit = ModelData(*stress_case_4_timed_deposit(self))

        self.stress_case_5_timed_expense = ModelData(*stress_case_5_timed_expense(self))


class ModelData:

    import numpy as np

    np.set_printoptions(precision=2, suppress=True)

    def __init__(self, info_array, moi, irr):
        self.info_array = info_array
        self.moi = moi
        self.irr = irr

    def __str__(self):
        return '{}\nMOI: {}\nIRR: {}%'.format(self.info_array, format(self.moi, '.2f'),
                                             format(100*self.irr, '.2f'))

    def get_info(self):
        return self.info_array, self.moi, self.irr


if __name__ == '__main__':

    model = Model(loan_amount=1000, loan_tenor=4, promo_rate=0.0, full_rate_2=.24, promo_period=0, amortization_type=1,
                  current_account_balance=1100, avg_deposit_3mth=1200, avg_weekly_expense_3mth=500, deposit_frequency=2,
                  deposit_only_stress=0.8, expense_only_stress=1.30, two_leg_deposit_stress=.85,
                  two_leg_expense_stress=1.15, timed_deposit_shock=0.75, timed_deposit_shock_weeks=2,
                  timed_expense_shock=1.25, timed_expense_shock_1_weeks=1, timed_expense_shock_2_weeks=3)

    # Printing results from running the model
    print('Base Loan CF:')
    print(model.base_loan_cf)

    print('\nAccount Balance Base Case:')
    print(model.account_balance_base_case)

    print('\nStress Case 1 (Deposit + Expense):')
    print(model.stress_case_1_deposit_expense)

    print('\nStress Case 2 (Deposit Only):')
    print(model.stress_case_2_deposit)

    print('\nStress Case 3 (Expense Only):')
    print(model.stress_case_3_expense)

    print('\nStress Case 4 (Timed Deposit):')
    print(model.stress_case_4_timed_deposit)

    print('\nStress Case 5 (Timed Expense):')
    print(model.stress_case_5_timed_expense)

