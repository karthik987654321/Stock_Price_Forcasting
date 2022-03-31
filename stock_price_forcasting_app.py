import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


st.write("""#  Stock Price Forcasting App""")

option = st.selectbox(
     'Select the company whose stock price you want to predict',
     ('maruti', 'infosys', 'bajaj'))

st.write('You selected:', option)

#read the dataframe
bajaj=pd.read_csv('bajaj_stock.csv')
maruti=pd.read_csv('maruti_stock.csv')
infy=pd.read_csv('infy_stock.csv')

a = "maruti"
b = "infosys"
c = "bajaj"

if option == a:
    #display the dataframe
    st.dataframe(maruti)  
    
    #display download option
    @st.cache
    def convert_df(df):
         # IMPORTANT: Cache the conversion to prevent computation on every rerun
         return df.to_csv().encode('utf-8')

    csv = convert_df(maruti)

    st.download_button(
         label="Download data as CSV",
         data=csv,
         file_name='maruti_df.csv',
         mime='text/csv',
         )
    
    st.metric(label="Opening price", value="7290.05", delta="136.4")
    
    #EDA
    maruti.set_index(pd.to_datetime(maruti['Date']),inplace=True)
    maruti=maruti.drop(columns=['Date','Turnover','Trades','Symbol','Series','Deliverable Volume','%Deliverble'],errors='ignore')

    
    #display line chart for closing price

    st.write("""##Closing price stock of Maruti""")

    st.line_chart(maruti.Close)

    st.write("""##Volume stock of Maruti""")

    st.line_chart(maruti.Volume)
    
    # Model Building
    X=maruti[['Open','High','Low']]
    Y=maruti['Close']
   
    
    # Spliting Dataset
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0)
    
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import confusion_matrix, accuracy_score
    regressor = LinearRegression()
    regressor.fit(X_train,Y_train)
    
    
    #forcasting 
    # Read dataset
    df1 = pd.read_csv("forcast_maruti.csv")
    df1.set_index(pd.to_datetime(df1['Date']),inplace=True)
    
    col = ['Open','High','Low']
    pred_input = df1[col]
    
    
    predicted=regressor.predict(pred_input)
    
    dfr= pd.DataFrame(predicted)

    st.write("""##   Forcasting 60 days future Closing Prices of Maruti Company Stock """)
    
    st.metric(label="Closing price", value="7425", delta="150.1131")


    # Displaying the results 
    st.dataframe(dfr)
    
    # Display download option
    @st.cache
    def convert_df(df):
         # IMPORTANT: Cache the conversion to prevent computation on every rerun
         return df.to_csv().encode('utf-8')

    csv = convert_df(dfr)

    st.download_button(
         label="Download data as CSV",
         data=csv,
         file_name='dfr.csv',
         mime='text/csv',
         )
    
    st.line_chart(dfr)

if option == b:
    #display the dataframe
    st.dataframe(infy)
    
    #display download option
    @st.cache
    def convert_df(df):
         # IMPORTANT: Cache the conversion to prevent computation on every rerun
         return df.to_csv().encode('utf-8')

    csv = convert_df(infy)

    st.download_button(
         label="Download data as CSV",
         data=csv,
         file_name='infy_df.csv',
         mime='text/csv',
         )
    
    st.metric(label="Opening price", value="1890", delta="2.25")
    
    #EDA
    infy.set_index(pd.to_datetime(infy['Date']),inplace=True)
    infy=infy.drop(columns=['Date','Turnover','Trades','Symbol','Series','Deliverable Volume','%Deliverble'],errors='ignore')
    
    #display line chart for closing price

    st.write("""##Closing price stock of Infosys""")

    st.line_chart(infy.Close)

    st.write("""##Volume stock of Infosys""")

    st.line_chart(infy.Volume)
    
    # Model Building
    X=infy[['Open','High','Low']]
    Y=infy['Close']
   
    
    # Spliting Dataset
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0)
    
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import confusion_matrix, accuracy_score
    regressor = LinearRegression()
    regressor.fit(X_train,Y_train)
    
    
    #forcasting 
    # Read dataset
    df2 = pd.read_csv("forcast_infy.csv")
    df2.set_index(pd.to_datetime(df2['Date']),inplace=True)
    
    col = ['Open','High','Low']
    pred_input = df2[col]
    
    
    predicted=regressor.predict(pred_input)
    
    dfr= pd.DataFrame(predicted)
    
    
    st.write("""##   Forcasting 60 days future Closing Prices of Infosys Company Stock """)
    
    st.metric(label="Closing price", value="1910", delta="-3.3559")

        
    # Displaying the results 
    st.dataframe(dfr)
    
    # Display download option
    @st.cache
    def convert_df(df):
         # IMPORTANT: Cache the conversion to prevent computation on every rerun
         return df.to_csv().encode('utf-8')

    csv = convert_df(dfr)

    st.download_button(
         label="Download data as CSV",
         data=csv,
         file_name='dfr.csv',
         mime='text/csv',
         )
    
    st.line_chart(dfr)


if option == c:
    #display the dataframe
    st.dataframe(bajaj)  
    
    #display download option
    @st.cache
    def convert_df(df):
         # IMPORTANT: Cache the conversion to prevent computation on every rerun
         return df.to_csv().encode('utf-8')

    csv = convert_df(bajaj)

    st.download_button(
         label="Download data as CSV",
         data=csv,
         file_name='bajaj_df.csv',
         mime='text/csv',
         )
    
    st.metric(label="Opening price", value="6882", delta="95.3")
    
    #EDA
    bajaj.set_index(pd.to_datetime(bajaj['Date']),inplace=True)
    bajaj=bajaj.drop(columns=['Date','Turnover','Trades','Symbol','Series','Deliverable Volume','%Deliverble'],errors='ignore')
    
    #display line chart for closing price

    st.write("""##Closing price stock of Bajaj""")

    st.line_chart(bajaj.Close)

    st.write("""##Volume stock of Bajaj""")

    st.line_chart(bajaj.Volume)
    
    # Model Building
    X=bajaj[['Open','High','Low']]
    Y=bajaj['Close']
   
    
    # Spliting Dataset
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0)
    
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import confusion_matrix, accuracy_score
    regressor = LinearRegression()
    regressor.fit(X_train,Y_train)
    
    
    #forcasting 
    # Read dataset
    df3 = pd.read_csv("forcast_bajaj.csv")
    df3.set_index(pd.to_datetime(df3['Date']),inplace=True)
    
    col = ['Open','High','Low']
    pred_input = df3[col]
    
    
    predicted=regressor.predict(pred_input)
    
    dfr= pd.DataFrame(predicted)
    
    
    st.write("""##   Forcasting 60 days future Closing Prices of Bajaj Company Stock """)
    
    st.metric(label="Closing price", value="7100", delta="132.431")

    
    # Displaying the results 
    st.dataframe(dfr)
    
    # Display download option
    @st.cache
    def convert_df(df):
         # IMPORTANT: Cache the conversion to prevent computation on every rerun
         return df.to_csv().encode('utf-8')

    csv = convert_df(dfr)

    st.download_button(
         label="Download data as CSV",
         data=csv,
         file_name='dfr.csv',
         mime='text/csv',
         )
    
    st.line_chart(dfr)
