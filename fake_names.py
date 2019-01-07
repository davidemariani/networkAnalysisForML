import random
import urllib.request
import pandas as pd


word_url = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
response = urllib.request.urlopen(word_url)
long_txt = response.read().decode()
words = long_txt.splitlines()

upper_words = [word for word in words if word[0].isupper()]

postfixes = ['Ltd', 'LTD', 'S.p.a.', 'Limited', 'Srl', 'GmbH', 'United', 'LLC', 'Inc.',
            'Co.', 'Corporation']

def rand_name():
    """
    This function creates a random name
    """
    name_1 = upper_words[random.randint(0, len(upper_words)-1)]+' '+upper_words[random.randint(0, len(upper_words)-1)]+' '+postfixes[random.randint(0, len(postfixes)-1)]
    name_2 = upper_words[random.randint(0, len(upper_words)-1)]+' '+postfixes[random.randint(0, len(postfixes)-1)]
    name = [name_1, name_2]
    return name[random.randint(0, len(name)-1)]

def fake_list(number):
    """
    This function creates a list of random names, checking that they are all different
    """
    names = set()
    count = 0
    
    while count < number:
        new_name = rand_name()
        if new_name not in names:
            names.add(new_name)
            count+=1
    return list(names)

def create_fake_names_dict(customers_name, debtors_name):
    """
    This function, given a list of customer names and a list of debtor names, creates two separate dictionaries (one for customer and one for debtors)
    to address each name to a fake one.
    """
    print("Creating the dictionary for fake names...")
    fake_names_cust_debt = fake_list(len(debtors_name) + len(customers_name))
    fake_names_cust = fake_names_cust_debt[:len(customers_name)]
    fake_names_debt = fake_names_cust_debt[len(customers_name):]

    customers_fake_dict = {}
    for i in range(len(customers_name)):
        customers_fake_dict[customers_name[i]]=fake_names_cust[i]

    debtors_fake_dict = {}
    for j in range(len(debtors_name)):
        debtors_fake_dict[debtors_name[j]]=fake_names_debt[j]

    return customers_fake_dict, debtors_fake_dict

def create_fn_pickle(df):
    """
    This function creates a dataframe of fake names and save it as pickle file in the current folder
    """

    #fake names module import
    from fake_names import rand_name, fake_list, create_fake_names_dict

    dicts = create_fake_names_dict(df['customer_name_1'].unique(), df['debtor_name_1'].unique())
    dict = pd.DataFrame({'customers':dicts[0], 'debtors': dicts[1]}) 
    print("Saving the file 'fake_names_dictionary.pkl' at the current folder...")
    dict.to_pickle("fake_names_dictionary.pkl")
    return dicts[0], dicts[1]



def generate_fake_names(df, from_scratch = False):
    """
    This function generates customers and debtors fake names, either generating them from scratch using create_fn_pickle
    or loading them from a predefined pickle file.
    """
    if from_scratch:
        #fake names module import
        customers_fake_dict, debtors_fake_dict = create_fn_pickle(df)
    else:
        print("Loading the fake names dictionary from fake_names_dictionary.pkl...")
        fake_dicts = pd.read_pickle('fake_names_dictionary.pkl')
        customers_fake_dict = fake_dicts['customers'].dropna().to_dict()
        debtors_fake_dict = fake_dicts['debtors'].dropna().to_dict()

    return customers_fake_dict, debtors_fake_dict
