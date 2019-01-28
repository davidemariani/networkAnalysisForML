#==================================================================================#
# Author       : Davide Mariani                                                    #  
# University   : Birkbeck College, University of London                            # 
# Programme    : Msc Data SCience                                                  #
# Script Name  : features_utils.py                                                 #
# Description  : utils for feature engineering                                     #
# Version      : 0.1                                                               #
#==================================================================================#
# This file contains several functions to add, process and enhance features        #
#==================================================================================#


def _xor0(x):
    """
    This function replaces nans with 0
    """
    return 0. if np.isnan(x) else x
xor0 = np.vectorize(_xor0)


def we_share(lst):
    """
    This function return the ratio of weekend payments for an instrument. nan if there's no weekend payment.
    """
    res = np.nan
    wec = 0
    datec = 0
    for x in lst:
        if not pd.isnull(x):
            datec+=1
            if x.weekday()>4:
                wec+=1
    if datec>0:
        res=wec/datec
    return res


def add_main_features(inst, impthr, imp2thr, prefix=''):
    """
    This function add the main features to an input instruments dataframe
    """
    impthr = 0.009 #threshold for impairments
    imp2thr = 0.04 #threshold for impairment_3

    #define the discharge loss as difference between invoice_amount and discharge amount...
    inst[prefix+"discharge_loss"] = xor0(inst[prefix+"invoice_amount"] - inst[prefix+"discharge_amount"])
    inst.loc[pd.isnull(inst[prefix+"discharge_amount"]), prefix+"discharge_loss"] = 0. #...but it is 0 for NaN discharge_amount

    #define the presence of impairment1 as deduction_amount>0.009
    inst[prefix+"has_impairment1"] =  inst.deduction_amount>impthr

    #define the presence of impairment2 as discharge_loss>0.009
    inst[prefix+"has_impairment2"] =  inst.discharge_loss>impthr

    #define the presence of impairment3 as discharge_loss>proportion of invoice amount or deduction_amount>proportion of invoice amount
    inst[prefix+"has_impairment3"] =  (inst.discharge_loss>imp2thr*inst.invoice_amount) | (inst.deduction_amount>imp2thr*inst.invoice_amount)

    #instrument which are open and more than 90 days past the due date 
    inst[prefix+"is_pastdue90"] =  inst.due_date.apply(lambda x: (ReportDate - x).days > 90) & (inst.document_status=="offen")

    #instrument which are open and more than 180 days past the due date
    inst[prefix+"is_pastdue180"] =  inst.due_date.apply(lambda x: (ReportDate - x).days > 180) & (inst.document_status=="offen")

    #instrument with prosecution
    inst[prefix+"has_prosecution"] = inst.prosecution.apply(lambda x: x=="Ja")

    #amount of the last payment for a certain instrument
    inst[prefix+"last_payment_amount"] = xor0(inst.payment_amount.apply(lambda x: x[-1]))

    #sum of all the distinct entries for a single instrument
    inst[prefix+"total_repayment"] = xor0(inst.payment_amount.apply(lambda x: sum(list(set(x))))) #sum of distinct entries

    #sum of discharge_loss and deduction_amount
    inst[prefix+"total_impairment"] = xor0(inst.discharge_loss) + xor0(inst.deduction_amount)

    #field indicating if an instrument is open or not
    inst[prefix+"is_open"] = inst.document_status.apply(lambda x: x=="offen")

    #sort instruments dataset by invoice date and debtor id
    inst = inst.sort_values(by=[prefix+"invoice_date", prefix+"debtor_id"], ascending=[True, True])

    inst[prefix+"we_payment_share"] = inst.payment_date.apply(lambda x: we_share(x))
    print("Weekend payment shares: {:}".format(inst.we_payment_share.value_counts()))

    #this indicates if an instrument has a purchase amount (if not, the client is not involved in repayment)
    inst[prefix+"has_purchase"] = inst.purchase_amount.apply(lambda x: x>0.009)

    #this indicates if an instrument has a deduction amount
    inst[prefix+"has_deduction"] = inst.deduction_amount.apply(lambda x: x>0.009)

    #this field indicates if an instrument is due
    inst[prefix+"is_due"] = inst.due_date.apply(lambda x: x < ReportDate)

    #discharge amount
    inst[prefix+"has_discharge"] = inst.discharge_amount>0.001