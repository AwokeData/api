import numpy as np


def stress_case_5_timed_expense(model):

    # Defining rows
    acct_bal_BOP = 0
    income = 1
    expenses = 2
    additional_expenses = 3
    loan_expense = 4
    loan_principal = 5
    acct_bal_EOP = 6
    CF = 7

    info_array = np.zeros((CF + 1, model.loan_tenor + 1))

    info_array[acct_bal_EOP, 0] = model.current_account_balance
    info_array[CF, 0] = - model.loan_amount

    for week in range(1, model.loan_tenor + 1):

        info_array[acct_bal_BOP, week] = info_array[acct_bal_EOP, week - 1]

        info_array[income, week] = model.account_balance_base_case.info_array[1, week]

        info_array[expenses, week] = model.account_balance_base_case.info_array[2, week]

        if week == model.timed_expense_shock_1_weeks or week == model.timed_expense_shock_2_weeks:
            info_array[additional_expenses, week] = model.avg_weekly_expense_3mth * (1 - model.timed_expense_shock)
        else:
            info_array[additional_expenses, week] = 0

        if info_array[:loan_expense, week].sum() < 0:
            info_array[loan_expense, week] = 0
        else:
            info_array[loan_expense, week] = - min(info_array[:loan_expense, week].sum(),
                                                   model.base_loan_cf.info_array[1:3, week].sum())

        if info_array[:loan_principal, week].sum() < 0:
            info_array[loan_principal, week] = 0
        else:
            info_array[loan_principal, week] = - min(info_array[:loan_principal, week].sum(),
                                                     model.base_loan_cf.info_array[3, week])

        info_array[acct_bal_EOP, week] = info_array[:acct_bal_EOP, week].sum()

        info_array[CF, week] = -info_array[loan_expense:acct_bal_EOP].sum()

    # MOI
    moi = info_array[CF, 1:].sum() / -info_array[CF, 0]

    # IRR in years, converted to weeks
    irr = (1 + np.irr(info_array[CF])) ** 52 - 1

    return info_array, moi, irr




