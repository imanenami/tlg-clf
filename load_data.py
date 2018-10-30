import pandas as pd
import calendar
import datetime
from sqlalchemy import create_engine, MetaData, Table, select

user = 'user'
passwd = 'pass'
server = 'localhost'
db = 'telegram'

engine = create_engine('mysql://{}:{}@{}:3306/{}?charset=utf8mb4'.format(user, passwd, server, db))
con = engine.connect()
metadata = MetaData()

tlgrphy_table = Table('telegraphy', metadata, autoload=True, autoload_with=engine)

start_date = datetime.datetime(2018, 8, 4)
start_date_unix = calendar.timegm(start_date.timetuple())

stmt = select([tlgrphy_table.columns.txtContent, tlgrphy_table.columns.label])
stmt = stmt.where(tlgrphy_table.columns.label > 0)
stmt = stmt.where(tlgrphy_table.columns.date >= start_date_unix)
stmt = stmt.where(tlgrphy_table.columns.channel != 'tlgrphy')

res = con.execute(stmt)
res_list = res.fetchall()

df = pd.DataFrame(res_list, columns=['text', 'label'])
df.to_excel('tlg_data.xlsx', encoding='utf8')
