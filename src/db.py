# --- Code Block ---
#create teradata connection

uda = teradata.UdaExec(appName='project_name', version='1.0')
connection = uda.connect(driver='Teradata', 
                 method='odbc', system='TDPROD', username=os.environ.get('user'), 
                 password=getpass.getpass(), authentication="LDAP")

# --- Code Block ---
'''#policies
pol_sql = open(path + "\SQL\Prod_Policies.sql",'r').read()
pol_sql = pol_sql.format(StartDate= '2022-01-01')
pol = pd.read_sql(pol_sql,connection)
pol.columns = pol.columns.str.lower()'''

# --- Code Block ---
#icd10
icd_sql = open(path + "\SQL\Train_ICD10CM.sql",'r').read()
icd_sql = icd_sql.format(StartDate= startdate)
icd = pd.read_sql(icd_sql,connection)

# --- Code Block ---
#cpt & drg
cpt_sql = open(path + "\SQL\Train_CPT.sql",'r').read()
cpt_sql = cpt_sql.format(StartDate= startdate)
cpt = pd.read_sql(cpt_sql,connection)

