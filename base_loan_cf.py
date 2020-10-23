import numpy as np


# [base_loan_cf] computes the info_array, moi, and irr for the Base Loan CF model

def base_loan_cf(model):

    # Defining rows
    loan_balance_BOP = 0
    interest_payment_1 = 1
    interest_accrual = 2
    principal_payment = 3
    balance_EOP = 4
    CF = 5

    # Initializing our array of information. It has [loan_tenor] + 1 columns representing weeks and 6 rows representing
    # Loan Balance BOP, Interest Payment 1, Interest Accrual, Principal Payment, Balance EOP, and CF, respectively.
    info_array = np.zeros((6, model.loan_tenor + 1))

    # Setting initial Balance EOP to [loan_amount] and initial CF to negative [loan_amount]
    info_array[balance_EOP, 0] = model.loan_amount
    info_array[CF, 0] = - model.loan_amount

    for week in range(1, model.loan_tenor + 1):

        info_array[loan_balance_BOP, week] = info_array[balance_EOP, week - 1]

        info_array[interest_payment_1, week] = info_array[loan_balance_BOP, week] * \
                                               ((1 + model.promo_rate) ** (7.0 / 365) - 1) \
                                               * int(week <= model.promo_period)

        info_array[interest_accrual, week] = info_array[loan_balance_BOP, week] * \
                                            ((1 + model.full_rate_2) ** (7.0 * model.loan_tenor / 365) - 1) \
                                            * int(week == model.loan_tenor and week > model.promo_period)

        if model.amortization_type == 1 or model.amortization_type == 'Balloon':
            info_array[principal_payment, week] = info_array[loan_balance_BOP, week] * int(week == model.loan_tenor)
        elif model.amortization_type == 2 or model.amortization_type == 'Linear':
            info_array[principal_payment, week] = float(model.loan_amount) / model.loan_tenor * \
                                                  int(week <= model.loan_tenor)
        elif model.amortization_type == 3 or model.amortization_type == 'Mortgage':
            info_array[principal_payment, week] = (np.pmt((1 + model.full_rate_2) ** (7.0 / 365) - 1,
                                                          model.loan_tenor - week + 1,
                                                   -info_array[loan_balance_BOP, week]) -
                                                   info_array[interest_payment_1:principal_payment, week].sum()) * \
                                                   int(week <= model.loan_tenor)

        info_array[balance_EOP, week] = info_array[loan_balance_BOP, week] - info_array[principal_payment, week]

        info_array[CF, week] = info_array[interest_payment_1:balance_EOP, week].sum()

    # MOI
    moi = sum(info_array[CF, 1:]) / - info_array[CF, 0]

    # IRR in years, converted to weeks
    irr = (1 + np.irr(info_array[CF])) ** 52 - 1

    return info_array, moi, irr
