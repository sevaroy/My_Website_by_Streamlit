import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
from PIL import Image
import os,glob
from zipfile import ZipFile
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib
import altair as alt
matplotlib.use('Agg')# To Prevent Errors

from sklearn.datasets import  load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

code = '''iris_df=load_iris()

iris_df.data
iris_df.target'''
st.code(code,language='python')



st.title('簡介與作品集')

st.text('This is some text.')

st.markdown('Streamlit is **_really_ cool**.')

st.write('Hello, *World!* :sunglasses:')

st.header('This is a header')

code = '''def hello():
    print("Hello,Streamlit!")'''
st.code(code,language='python')

chart_data = pd.DataFrame(
	np.random.randn(20,3),
	columns=['a','b','c'])

st.line_chart(chart_data)
st.area_chart(chart_data)


chart_data = pd.DataFrame(
	np.random.randn(50,3),
	columns=['a','b','c'])
st.bar_chart(chart_data)

arr = np.random.normal(1,1,size=10000)
plt.hist(arr,bins=20)

st.pyplot()

df1 = pd.DataFrame(
	np.random.randn(200,3),
	columns = ['a','b','c'])

st.vega_lite_chart(df1,{
	'mark': 'circle',
	'encoding': {
	 		'x': {'field': 'a', 'type': 'quantitative'},
	 		'y': {'field': 'b', 'type': 'quantitative'},
	 		'size': {'field': 'c', 'type': 'quantitative'},
	  		'color': {'field': 'c', 'type': 'quantitative'},
	   },
})

df2 = pd.DataFrame(
	np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])
df2

if st.button('Say Hello'):
	st.write('why hello there')
else:
	st.write('Goodbye')


genre = st.radio(
	"write's your favorite movie genre",
	('comedy','Drama','Documentary'))

if genre =='comedy':
	st.write('You selected Comedy.')
else:
	st.write("you didn't select comey")

st.info('This is a purely informational message')
st.warning('This is a warning')
st.success('This is a success message!')


df = pd.DataFrame(np.random.randn(200, 3), columns=["a", "b", "c"])

c = alt.Chart(df).mark_circle().encode(x="a", y="b", size="c", color="c").interactive()

st.title("These two should look exactly the same")

st.write("Altair chart using `st.altair_chart`:")
st.altair_chart(c)

st.write("And the same chart using `st.write`:")
st.write(c)



"""
## 我的自傳

(1) 成長背景

我是黃翊鈜，土生土長桃園人，畢業於台中私立逢甲大學運輸科技與管理學系，熱愛自助旅行，藉由旅遊更加了解各國歷史文化，平常休閒活動為電玩以及戲劇研究。

(2) 求學歷程

我從小熱愛電腦，大學時期自學拆解主機升級配備，學會簡易故障排除，同時期接觸了Ubuntu，對相對於Windows的Linux產生了興趣，大學除了本科系課程外，自己選課多為資訊課程居多，大學專題選擇製作簡易交通問卷資料庫，增加對資料庫的概念。

(3) 工作經歷

退伍後，在產物保險公司擔任營業人員，在工作期間了解到產物保險的營業通路作業，以及汽車保險的核保理賠，獲益良多。

之後因朋友邀請結伴到澳洲打工度假，真正學會料理自己的生活，增進了英語口說能力，在昆士蘭最大農場企業工作兩年，擔任香蕉包裝生產線工作人員，非常忙綠，香蕉串還在樹上的時候到切割下來包裝進紙箱的過程都有參與，這個過程讓我了解到生產線的基本概念，在場內擔任要職的我，上班除了薪水非常不錯以外也得到成就感。

回國後，進入EGAT擔任資材部門的物料管理人員，原本對資材物料及庫存管理一竅不通的我，遇到問題時都會記下來製作成文件，讓後進人員下次遇到相同問題時，可以快速得到解決辦法，讓作業流程不再因為每個人作業方式的不同而產生瓶頸。工作一陣子後，原本對航空器不熟的我也能背出幾款熱門發動機以及其品牌公司，對空中巴士以及波音製作的航空器更是能做出說明及比較，不知不覺之中儼然是個業餘航空迷。

在經過幾番考慮後，對於未來的徬徨依舊，我知道自己始終還是要走資訊人這條路，所以我報名了中壢資策會的AI/BigData 巨量資料工程師養成班，在這半年的時間，我找回了那個熱愛電腦資訊的自己，也正式踏入了軟體工程的行列，一條持續學習的路。

201807~201801 中壢資策會進修時段
資策會結訓證書 https://reurl.cc/YlAljO

後來進入Gamamobi(現地心引力)擔任機器學習工程師，主要工作內容是協助其他部門，例如廣告投放以及遊戲營運部門利用tableau進行資料視覺化，從中產生見解。在手機遊戲代理的營運中，廣告行銷預算占比大，評估與設計協助建立營銷資料倉儲協助營運部門了解市場趨勢。

完成的工作項目：
1.分析廣告投放後反饋回來的資料進行使用者分布分析以及廣告渠道所帶來的營收分佈，在會議上用即時回饋做出視覺化的方式，回答市場部的問題。
2.將使用者數據進行RFM客戶細分，藉此了解使用者付費情形，並提供行銷部門針對不同客群進行訊息推播
3.整合其他數據進行預處理
4.串接Facebook ads,Google ads,GA之數據至Dashboard，提供相關部門更清楚快速的視覺化方案
5.大量分析廣告素材之數據，找出熱門素材特性
6.對現有資料進行資料分析
7.使用Python進行機器學習建模分析與評估

使用工具：Python,Machine Learning, tableau desktop, tableau prep builder, tableau server,Data Studio,Facebook ads,Google ads, GA, BigQuery


(4) 技能與應用

- Hadoop Ecosystem:Hadoop knowledge, Cloudera, Hortonworks, Spark
- OS:MacOS,CentOS,Ubuntu.
- General: Python, Amazon Web Service, Google Cloud,
- Data warehouse: Bigquery,HDFS,AWS S3,GCS
- Data manipulation: Pandas, SQL,Tableau Prep Builder
- Machine learning: Scikit-learn, BQ ML, AUTO ML.
- Deep Learning:Keras,PyTorch.
- Visualization: Tableau Desktop,Kibana,Seaborn
- Application: Line developer
- Database: Elasticsearch,ELK
- Front end:streamlit
- Devops:Docker
- version control:Git

(5) 未來展望

未來五年之進修計劃

欲考取之證照：

AWS 雲端從業人員 
Google Cloud for Data Engineering
Google Cloud for Cloud architecture
AWS Certified Data Analytics-Specialty
AWS Certified Machine Learning Specialty


在四月參加了百日馬拉松-機器學習，增加自己的資料科學知識，期望自己不只能夠成為資料工程師，也能參與資料科學家的工作。

證書：https://reurl.cc/pDkD1r

為了增加自己coding能力，我也訂閱了Datacamp線上課程做為加強課程，訂閱已邁入第二年，豐富的課程外加專業的類別，讓我能更容易建立起專業技能與概念。

由於在網路爬蟲部分的理解較為不足，在十一月報名參加百日馬拉松 網路爬蟲線上課程，藉以加強自己的Data Collection的能力。

未來希望自己在工作餘力之際，能夠加強自己的英文能力(新制多益535)，尤其是考取托福成績，取得成績後，能夠報名伊利諾大學在Coursera開設的資料科學碩士學位增進自己的能力，這個課程是個線上課程，在本地學習，結業可至美國伊利諾大學參加畢業典禮，期望自己能夠達成這項成就，讓自己的學經歷更完整。


"""

