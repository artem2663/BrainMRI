import pypyodbc as db
import pandas as pd

class DBManager:

    def __init__(self):
        self.cnxn = db.connect("Driver={SQL Server Native Client 11.0};"
                               "Server=localhost\***;"
                               "Database=***;"
                               "uid=***;pwd=***")

    def GetPatientsByCode(self, codes):
        sql = "EXEC prcPatientDataGetByCode @Codes='" + codes + "'"
        df = pd.read_sql_query(sql, self.cnxn)
        return df


