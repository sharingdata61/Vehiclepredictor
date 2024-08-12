import streamlit as st
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from prediction import predict
from datetime import datetime, time
import joblib

model = joblib.load('Tripmodel.joblib')
label_encoder = joblib.load('LabelEncoder.joblib')
label_encoder1 = joblib.load('LabelEncoder1.joblib')

st.title("Vehicle Trip Predictor")
st.markdown("ML model to predict Trip Duration")

st.header("Predictor")

# def fetch_routes_from_db():
#     cnxn_str = ("Driver={ODBC Driver 17 for SQL Server};"
#             "Server=10.20.30.75,1433;"
#             "Database=BI_TEST;"
#             "UID=HC_Gaurav4;"
#             "PWD=Gaurav4;"
#             )


#     engine = create_engine('mssql+pyodbc:///?odbc_connect=' + cnxn_str)


#     query = """SELECT DISTINCT Route_Name FROM VehicleTrip_Detail_Process_Data WHERE Route_Name NOT LIKE '%STL%' """
 
#     df = pd.read_sql(query, engine)


#     routes = df['Route_Name'].tolist()
#     return routes


# routes1 = fetch_routes_from_db()
# print(routes1)

routes1 = [
    "102ADown","102Down","102Up","104Down","106Down","108AUP","112AUp","113Down","113Up","114A_Down",
    "114A_Up","114CDown","116AUp2","116DOWN","120ADown","120Down","125ExtDown","126Up","128Up","129Down","129Up","133Down","135Down",
    "137Down","137Up","144A_Up","144Up","147Down","149BDown","161AUp","161Up","164ADown","165AUp","171Down",
    "171Up","172ADown","172BUP","173Up","174Down","179Down","182ACLDOWN","191Down","193Down","193Up","194Down","206ADown",
    "207Down","215Up","217Up","221DOWN","221UP","227Down","232Down","237ADown","237AUp","237Up","239Down","239Up","254ADown","258DOWN",
    "259UP","270Down","274ADown","281ADOWN","281AUP","307Down","307Up","320Up","359Down","375DOWN","390Down","391Down","391Up","405ADOWN",
    "410CLUp","412Up","413Down","419_Up","419AUP","425CLUp","429ADown","429CLDown","448CLUP","460CLDOWN","473EDOWN","47ACLUP",
    "47ALinkDown","47ALinkUP","507CLDown","507EUP","511DOWN","511UP","518Up","522BDown2","522CLDOWN","523Down","539ADown","539UP","544DOWN",
    "548CLUP",    "567AUp2",    "567Up",      "568ADown",   "568Down",    "568Up",      "569AUP",    "569DOWN",
    "578A_Up",    "588DOWN",    "588UP",      "615UP",      "623DOWN",    "623UP",     "708_Down",
    "709ADOWN",   "712Up",      "717CDown3",  "717CUp2",    "717CUp3",    "717Up2",    "719ADOWN",
    "721BDown",   "721UP",      "722UP",      "724UP",      "727DOWN",    "728ADown",  "728EUp",
    "745ADown",   "752Up",      "761DOWN",    "763ADown",   "764UP",      "770ADOWN",  "770AUp",
    "772Down",    "783AUp2",    "783Down",    "794ADown2",  "794AUp2",    "801AUp",    "806Down",
    "807AUp",     "808STDOWN",  "810DOWN",    "810UP",      "816BDown",   "817Down",   "817NDown",
    "818AUp",     "821ADown",   "821Up",      "822UP",      "825UP",      "828EDown",  "828UP",
    "829Up",      "833ADown",   "834Down",    "836DOWN",    "836SPLDown", "845DOWN",   "850Down",
    "850LTDDown", "858Down",    "858Up",      "85ExtDown",  "861AUP",     "871UP",     "874AUp2",
    "887AUp2",    "887Down",    "892Down",    "893ADown",   "893DOWN",    "901AUp",    "901CLUP",
    "913Down",    "917Up",      "918DOWN",    "921AExtUP",  "926UP",      "928Down",   "928UP",
    "929UP",      "938ACLDown", "938ACLUp",   "938BDown",   "939Down",    "940ADown",  "940AUp",
    "944Down",    "944Up",      "945AUP",     "949AUp",     "949Down",    "954UP",     "956Down","957UP","966AUp",
    "966Up",       "972AExtDown", "978DOWN",    "978UP",      "980Down",    "981AUp",    "989AUp",
    "990B_Down",   "990CLUP",     "992Down",    "998Down",    "AIIMS(B)-NGT", "AIR-06Down", "AirportExp04AUP",
    "AirportExp04Down", "BPExpUP",    "CBD2(+)Up",  "Central_Secretariat_Circular_Service",
    "Central_Secretariat_Link_Service_PM_Sangrahalaya", "DW3ADOWN", "DW3Up", "DW4AUp",
    "GL23Down",    "GL23Up",      "GL91ADown",  "MC715Down",  "NGT-AIIMS(B)", "OMS(-)",     "OMSEVPlus1",
    "PM_Sangrahalaya_Link_Service_Central_Secretariat", "RL75ADown", "RL75Down",  "RL77BDOWN", "RL-79UP",
    "TMS(-)DK-KG", "TMS(+)KashmereGate", "TMS(+)KG-DK", "102AUp", "103Down", "104Up", "106A_Down",
    "107Down",     "108ADOWN",    "108UP",      "109ADown",   "109Down",    "113ADown",   "114ADown",
    "114B_UP",     "114Down",     "114Up",      "116ADown2",  "123Down",    "125ExtUp",   "130AUp",
    "130Up",       "133Up",       "135Up",      "136ADown",   "137AUp",     "144A_Down",  "144Down",
    "148AUp",      "148Up",       "158Down",    "164AUp",     "165A_Up",    "169DOWN",    "169UP",
    "172Down",     "172Up",       "175Up",      "181CLDown",  "182BDown",   "188Down",    "191ADown2",
    "191Up",       "193AUp",      "207Up",      "213Down",    "214CLUP",    "215ADown",   "227ADOWN",
    "236DOWN",     "236EUp",      "236UP",      "247Down",    "247EUp",     "247UP",      "253DOWN",
    "254AUp",      "260DOWN",     "261Up",      "267Down",    "270Up",      "274Up",      "317Down",
    "334Up",       "340Down",     "340Up",
        "359Up",        "370Down",      "370UP",       "380Up",       "39UP",        "405DOWN",     "405UP",
    "410CLDown",    "412Down",      "413Up",       "425CLDown",   "427_Down",    "427_Up",      "434Up",
    "456Down",      "456Up",        "460CLUP",     "473CLUP",     "47ACLDOWN",   "505Up",       "523Up",
    "534UP",        "539Ext_Up",    "567ADown2",   "567BUp2",     "567Down",     "569UP",       "578A_Down",
    "578CDown",     "578EUp",       "604eUp",      "604SPLDown",  "605UP",       "621UP",       "623EXTUP",
    "624ACLUP",     "628Up",        "701Down",     "702ADown",    "702AUp",      "702Down",     "703ADown",
    "708_Up",       "708UP",        "711UP",       "712Down",     "715Down",     "717ADOWN",    "717Down2",
    "717EDown",     "721DOWN",      "724ADown",    "724ADown2",   "724AUp",      "724AUp2",     "724BDown3",
    "727UP",        "728EDown",     "729BDown",    "729EDown2",   "729Ext_Down", "729Ext_Up",   "740ExtDOWN",
    "743BDown",     "743BUp",       "745DOWN",     "761UP",       "764DOWN",     "770ALTDDown", "774BUp",
    "775EUp",       "778DOWN",      "778UP",       "780DOWN",     "780UP",       "781Down2",    "783ADown2",
    "784ADown",     "790+825Up",    "790+876UP",   "792AUp",      "794UP",       "801Down",     "807BDown",
    "807BUp",       "816Up",        "817Up",       "825+790Down", "825DOWN",     "826Down",     "828ADown",
    "833AUp",       "835BDown",     "835BUp",      "836578LTD_Up","836SPLUp",    "838DOWN",     "844EDown",
    "844EUP",       "845UP",
    "849Up",        "850Up",        "859ADown",     "85ExtUp",     "861ADOWN",    "863AUp2",     "871DOWN",
    "872Up",        "874DOWN",      "874UP",        "879UP",       "883ExtDown",  "883UP",       "885Down",
    "885EUp",       "886UP",        "887ADown2",    "88AUp",       "901BDown",    "901CLDOWN",   "908DOWN",
    "913Up",        "917ADOWN",     "917AUP",       "917Down",     "921AExtDown", "921ExtDown",  "923A_Up",
    "923Up",        "928ADown",     "928AUp",       "929DOWN",     "929EUp",      "934Down",     "934Up",
    "939Up",        "962Up",        "964Up",        "972EUP",      "979AUp",      "979Down",     "980Up",
    "981Down",      "982AUp",       "982UP",        "989Up",       "990AUP",      "990E_CL_UP",  "991UP",
    "993Up",        "AirportExp04ADOWN", "AirportExp08Down", "AirportExp08Up", "BPExpADown", "DW4Down",    "DW5Down",
    "DW5Up",        "GL91Down",     "ML06Up",       "OMS(+)",      "OMS+SK-UN-UP","RL75AUp",     "RL77BUP",
    "RL-77DOWN",    "RL-77EXTUP",   "YMSMinus",     "YMSPlus",     "(-)OuterMudrika","103Up",    "106A_Up",
    "106Up",        "107Up",        "109Up",        "112ADown",    "112Up",       "113AUp",      "114B_Down",
    "114BDown",     "114BUp",       "114CUp",       "116UP",       "123Up",       "126Down",     "130ADown",
    "131ADown2",    "131Down",      "136AUp",       "136Up",       "139ADown",    "139Down",     "139Up",
    "141Down",      "143DOWN",      "143UP",        "148Down",     "149BUp",      "149Up",       "156Down",
    "156Up",        "159UP",
    "164Up",        "165A_Down",    "165ADown",     "165UP",       "172BDown",    "173Down",     "181AExtDown",
    "182ACLUP",     "188Up",        "193ADown",     "205DOWN",     "205UP",       "211DOWN",     "211UP",
    "214CLDOWN",    "217Down",      "218Up",        "227AUP",      "236EDown",    "253UP",       "258UP",
    "260UP",        "267Up",        "274AUp",       "274Down",     "281Up",       "310UP",       "311ADown",
    "317Up",        "334Down",      "336ADown",     "336AUp",      "375UP",       "378Down",     "378Up",
    "380Down",      "390Up",        "39DOWN",       "403CLDown",   "403CLUp",     "405AUP",      "410BDown",
    "419ADOWN",     "422Down",      "429CLUp",      "433CLUp",     "448CLDown",   "469UP",       "502Down",
    "507CLUp",      "511AUP",       "534DOWN",      "534EDown",    "534EUP",      "539DOWN",     "539Ext_Down",
    "540ACLUp",     "540CLDown",    "544UP",        "567BDown2",   "568AUp",      "569ADown",    "578DOWN",
    "578UP",        "604eDown",     "605ADown",     "605AUP",      "605DOWN",     "610SPL2Up",   "621DOWN",
    "623EXTDN",     "628Down",      "706AUP",       "706Down",     "706UP",       "709AUP",      "709UP",
    "715Up",        "717AUP",       "717CDown2",    "717EUp",      "717UP",       "722Down",     "725Down",
    "729BUp",       "729UP",        "73ADown",      "73AUP",       "740ExtUP",    "741Down",     "741Up",
    "751Up",        "752Down",      "753DOWN",      "763AUp",      "770CDown",    "773Down",     "773Up",
    "774BDown",     "774Down",
    "775EDown",     "781UP",        "783Up",        "784AUp",      "784Down",     "784Up",       "790ADOWN",
    "790ADown2",    "790AUP",       "790BUp",       "790UP",       "792Down",     "794Down",     "817ExtDown",
    "818ADown",     "818BDown",     "818BUp",       "819Up",       "821AUp",      "821Down",     "824Down",
    "828AUp",       "828EUp",       "834Up",        "835UP",       "836EDown",    "836EUp",      "838ADown2",
    "838AUp2",      "847AUp",       "847UP",        "849Down",     "850BDown",    "850LTDUp",    "859AUp",
    "85AExtUp",     "85DOWN",       "85UP",         "863ADown2",   "876+790Down", "878_Down",    "879AUP",
    "883B_LTD_Down","883DOWN",      "883ExtUp",     "885Up",       "886ADown",    "886Down",     "88ADown",
    "892Up",        "908UP",        "918UP",        "919Down",     "926AUp",      "937ADOWN",    "938BUP",
    "939ADown",     "939AUp",       "940Down",      "942AUp2",     "943Down",     "943Up",       "945ADOWN",
    "946Down",      "946uP",        "949ADown",     "949Up",       "954DOWN",     "956Up",       "957BDown",
    "957DOWN",      "961Down",      "961Up",        "962ADown",    "962AUp",      "962Down",     "964Down",
    "966ADown",     "972DOWN",      "972EXTDOWN",   "972EXTUP",    "972UP",       "978A_Down",   "978LTDUp",
    "979ADown",     "979Up",        "981ADown",     "982ADown",    "985_Down",    "985_Up",      "988Down",
    "988Up",        "990CLDOWN",    "990ECLUP",     "993Down",     "BPExpDown",   "CBD2(-)Down", "DW1Up",
    "DW3AUP",       "DW3Down",
    "DW4Up",        "GL91Up",       "MC127Down",    "MC127Up",     "MC137Down",   "MC137Down2",  "MC341Up",
    "ML96Down",     "OMS-SK-UN-Down","OMS-UN-SK-Down","RL79ADown",   "RL79AUp",    "RL-79DOWN",   "Supreme_Court_Circular_Sewa",
    "TMS(+)DK-KG",  "(+)OuterMudrika","103EDown",   "103EUp",      "108DOWN",     "109AUp",      "112Down",
    "114AUp",       "120AUp",        "120Up",       "128Down",     "130Down",     "131AUp2",     "131Up",
    "136Down",      "137ADown",      "139AUp",      "140DOWN",     "140UP",       "141Up",       "147Up",
    "148ADown",     "149Down",       "153Down",     "153Up",       "158Up",       "159DOWN",     "161ADown",
    "161Down",      "164Down",       "165DOWN",     "172AUp",      "172EDown",    "172EUP",      "174Up",
    "175Down",      "179Up",         "181AExtUP",   "181CLUp",     "182BUp",      "191AUp2",     "194Up",
    "206AUp",       "213Up",         "215AUp",      "215Down",     "218Down",     "227Up",       "232Up",
    "236BDown",     "236BUp",        "237Down",     "246CLDown",   "246CLUp",     "247EDown",    "254Down",
    "254Up",        "259Down",       "261Down",     "281Down",     "309EXTUP",    "310DOWN",     "311AUp",
    "320Down",      "348Down",       "348Up",       "410BUP",      "419_Down",    "422UP",       "429AUp",
    "433CLDown",    "434Down",       "469DOWN",     "473CLDN",     "502UP",       "507EDOWN",    "511ADOWN",
    "518Down",      "522BUp",        "522CLUP",     "539AUp",      "540ACLDown",  "540CLUp",     "546Down",
    "546Up",        "548CLDOWN",
    "615DOWN",      "701Up",        "702Up",        "703AUp",      "703Down",     "703Up",       "706ADown",
    "708DOWN",      "709DOWN",      "711DOWN",      "717DOWN",     "719AUP",      "719DOWN",     "719UP",
    "721BUP",       "724BUp3",      "724DOWN",      "725Up",       "728AUp",      "729DOWN",     "729EUp2",
    "73DOWN",       "73UP",         "740DOWN",      "740UP",       "745UP",       "751Down",     "753UP",
    "763Down",      "763Up",        "770ALTDUp",    "770CUp",      "772Up",       "774Up",       "776Down",
    "776Up",        "781DOWN",      "781Up2",       "790AUp2",     "790BDown",    "790DOWN",     "792ADown",
    "792Up",        "801ADown",     "801Up",        "806Up",       "807ADown",    "816BUp",      "816Down",
    "817ExtUp",     "817NUp",       "819Down",      "822DOWN",     "824Up",       "826Up",       "828DOWN",
    "829Down",      "835DOWN",      "836578LTD_Down","836UP",      "838UP",       "844Down",     "844Up",
    "847ADown",     "847DOWN",      "850BUp",       "859Down",     "859Up",       "85AExtDown",  "863Down2",
    "863Up2",       "872Down",      "874ADown2",    "878_Up",      "879ADown",    "879DOWN",     "885EDown",
    "886AUP",       "887Up",        "890Down",      "890Up",       "891DOWN",     "891UP",       "893AUp",
    "893UP",        "901ADown",     "901BUp",       "919Up",       "921ExtUp",    "923A_Down",   "923Down",
    "926ADown",     "926DOWN",      "929EDown",     "937AUP",      "940Up",       "942ADown2",   "942Down",
    "942UP",        "957BUp",
    "966Down",      "971DOWN",      "971UP",        "972AExtUp",   "978A_Up",     "981Up",       "982DOWN",
    "989ADown",     "989Down",      "990ADOWN",     "990B_UP",     "990E_CL_DN",  "990ECLDOWN",  "991DOWN",
    "992Up",        "998Up",        "AIR-06Up",     "AirportExp04Up","BPExpAUP",   "DW1Down",     "DW4ADown",
    "GL91AUp",      "MC137Up",      "MC137Up2",     "MC341Down",   "MC715Up",     "ML06Down",    "ML96Up",
    "OMS+UN-SK-UP", "OMSEVMinus",    "RL75Up",       "RL-77EXTDOWN","RL-77UP",     "S1Down",      "S1Up",
    "TMS(-)KashmereGate", "TMS(-)KG-DK", "TMS(Plus)KashmereGate"
]


col1, col2, col3 = st.columns(3)
with col1:
    #st.text("Enter ")
    Day = st.selectbox("Select Day", [""] + ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

with col2:
     Route = st.selectbox("Select Route", [""] + routes1)
# st.button("Predict-Genre")
with col3:
    #st.text("Enter Age")
    Time = st.text_input("Enter Time in 24 Hour format","")

time_with_seconds = Time + ":00"

# def combine_time_with_date(time_val):
#     return datetime.combine(datetime.today(), time_val)

def time_to_float(t):
   return t.hour + t.minute / 60 + t.second / 3600


def convert_time_to_datetime(time_str):
    try:
        return datetime.strptime(time_str, "%H:%M:%S").time()
    except ValueError:
        return None

time_as_time_object = convert_time_to_datetime(time_with_seconds)

if time_as_time_object is None and Time:
    st.error("Invalid time format. Please use HH:MM format.")


input_data = pd.DataFrame(data=[[Day, Route, time_as_time_object]], columns=['Operated_Day', 'Route_Name', 'Actual_Trip_START'])
# input_data['StartTime'] = input_data['StartTime'].apply(combine_time_with_date)



# if gender in ['Male', 'Female']:
#     input_data[categorical_column] = label_encoder.transform([str(input_data[categorical_column][0])])[0]
# else:
#     st.warning("Please select a valid gender ('Male' or 'Female').")

# input_data[categorical_column] = label_encoder.transform(input_data[categorical_column][0])

if time_as_time_object:
    input_data['Actual_Trip_START'] = input_data['Actual_Trip_START'].apply(time_to_float)

    print(input_data['Actual_Trip_START'])

def float_to_time(f):
    hours = int(f)
    minutes = int((f - hours) * 60)
    seconds = int((f - hours - minutes / 60) * 3600)
    return time(hours, minutes, seconds)


if st.button("Predict"):
    if Route in routes1:
        # input_data['Actual_Trip_START'] = input_data['Actual_Trip_START'].apply(lambda x: x.timestamp())
        input_data['Operated_Day'] = label_encoder.transform([str(input_data['Operated_Day'][0])])
        input_data['Route_Name'] = label_encoder1.transform([str(input_data['Route_Name'][0])])
        print(input_data)
        result = predict(input_data)
        y_pred_datetime = [float_to_time(ts) for ts in result]
        
        st.markdown(y_pred_datetime[0])
        print(y_pred_datetime[0])
    else:
        st.warning("Please select a valid Route.")



# predicted_result = encoder.inverse_transform(result)