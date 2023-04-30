import re
import nltk
import string
import emoji
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import seaborn as sn
import pandas as pd
from urlextract import URLExtract
from collections import Counter
from wordcloud import WordCloud
from urllib.parse import urlparse
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def extract_datetime(text):
	text = text.split(' ')
	date, time = text[0], text[1]
	time = time.split('-')
	time = time[0].strip()

	return date + " " + time

def get_string(text):
	return text.split('\n')[0]

def preprocess(selected_pattern, data):
	patterns = ['\d{1,2}.\d{1,2}.\d{2,4}\s\d{1,2}:\d{2}\s-\s', '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s']
	if selected_pattern == '01.01.2023 12:24':
		pattern = patterns[0]
	else:
		pattern = patterns[1]

	messages = re.split(pattern, data)[1:]
	dates = re.findall(pattern, data)

	df = pd.DataFrame({'Kullanıcı Mesajı': messages, 'Mesaj Tarihi': dates})
	df['Mesaj Tarihi'] = df['Mesaj Tarihi'].apply(extract_datetime)

	users = []
	messages = []

	for message in df['Kullanıcı Mesajı']:
		entry = re.split('([\w\W]+?):\s', message)
		if entry[1:]:
			users.append(entry[1])
			messages.append(entry[2])

		else:
			users.append('Grup Bildirimi')
			messages.append(entry[0])

	df['Kullanıcı'] = users
	df['Mesajlar'] = messages
	df['Kullanıcı Mesajı'] = df['Mesajlar'].apply(get_string)

	df = df[['Kullanıcı Mesajı', 'Mesaj Tarihi', 'Kullanıcı']]

	df['Yıl'] = pd.to_datetime(df['Mesaj Tarihi']).dt.year
	df['Ay'] = pd.to_datetime(df['Mesaj Tarihi']).dt.month_name()
	df['Gün Ad'] = pd.to_datetime(df['Mesaj Tarihi']).dt.day_name()

	pazartesi = df['Gün Ad'] == 'Monday'
	sali = df['Gün Ad'] == 'Tuesday'
	carsamba = df['Gün Ad'] == 'Wednesday'
	persembe = df['Gün Ad'] == 'Thursday'
	cuma = df['Gün Ad'] == 'Friday'
	cumartesi = df['Gün Ad'] == 'Saturday'
	pazar = df['Gün Ad'] == 'Sunday'

	df.loc[pazartesi, 'Gün Ad'] = 'Pazartesi'
	df.loc[sali, 'Gün Ad'] = 'Salı'
	df.loc[carsamba, 'Gün Ad'] = 'Çarşamba'
	df.loc[persembe, 'Gün Ad'] = 'Perşembe'
	df.loc[cuma, 'Gün Ad'] = 'Cuma'
	df.loc[cumartesi, 'Gün Ad'] = 'Cumartesi'
	df.loc[pazar, 'Gün Ad'] = 'Pazar'

	ocak = df['Ay'] == 'January'
	subat = df['Ay'] == 'February'
	mart = df['Ay'] == 'March'
	nisan = df['Ay'] == 'April'
	mayis = df['Ay'] == 'May'
	haziran = df['Ay'] == 'June'
	temmuz = df['Ay'] == 'July'
	agustos = df['Ay'] == 'August'
	eylul = df['Ay'] == 'September'
	ekim = df['Ay'] == 'October'
	kasim = df['Ay'] == 'November'
	aralik = df['Ay'] == 'December'

	df.loc[ocak, 'Ay'] = 'Ocak'
	df.loc[subat, 'Ay'] = 'Şubat'
	df.loc[mart, 'Ay'] = 'Mart'
	df.loc[nisan, 'Ay'] = 'Nisan'
	df.loc[mayis, 'Ay'] = 'Mayıs'
	df.loc[haziran, 'Ay'] = 'Haziran'
	df.loc[temmuz, 'Ay'] = 'Temmuz'
	df.loc[agustos, 'Ay'] = 'Ağustos'
	df.loc[eylul, 'Ay'] = 'Eylül'
	df.loc[ekim, 'Ay'] = 'Ekim'
	df.loc[kasim, 'Ay'] = 'Kasım'
	df.loc[aralik, 'Ay'] = 'Aralık'

	return df

def stats(selected_user, df):
	extract = URLExtract()

	if selected_user != 'Genel':
		df = df[df['Kullanıcı'] == selected_user]

	num_messages = df.shape[0]
	words = []
	for message in df['Kullanıcı Mesajı']:
		words.extend(message.split())

	media_info = df[df['Kullanıcı Mesajı'] == '<Medya dahil edilmedi>']

	links = []
	for message in df['Kullanıcı Mesajı']:
		links.extend(extract.find_urls(message))

	emojis = []
	for message in df['Kullanıcı Mesajı']:
		emojis.extend([c for c in message if c in emoji.distinct_emoji_list(c)])

	return num_messages, len(words), media_info.shape[0], len(links), len(emojis)

def user_stats(df):
	df = df[df['Kullanıcı'] != 'Grup Bildirimi']
	count = df['Kullanıcı'].value_counts()

	percentages = (df['Kullanıcı'].value_counts() / df.shape[0]) * 100
    
	newdf = pd.DataFrame(percentages)
	newdf.columns.values[0] = 'Yüzde'
	
	return count, newdf

def word_freq(selected_user, df):
	freq_df = df[df['Kullanıcı'] == selected_user]
	freq_words = freq_df['Kullanıcı Mesajı'].tolist()
	freq_words = [i.lower() for i in freq_words]
	freq_punc = []
	  
	for o in freq_words:
		freq_punc += nltk.word_tokenize(o)
	stop_words = set(stopwords.words('turkish'))
	freq_punc = [o for o in freq_punc if o not in stop_words]  
	freq_punc = [o for o in freq_punc if o not in string.punctuation]
	freq_freq = Counter(freq_punc)
	freq_top = freq_freq.most_common(15)

	return freq_top

def print_wordcloud(dict_top, df):
	dict_top = dict(dict_top)
	wordcloud = WordCloud(width=350, height=350, background_color='white', min_font_size=5).generate_from_frequencies(dict_top)

	plt.figure(figsize=(5,9), facecolor=None)
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.tight_layout(pad=0)
	plt.show()

def emoji_stats(selected_user, df):
	if selected_user != 'Genel':
		df = df[df['Kullanıcı'] == selected_user]

	emojis = []
	for message in df['Kullanıcı Mesajı']:
		emojis.extend([c for c in message if c in emoji.distinct_emoji_list(c)])

	emojidf = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
	emojidf.rename(columns={0:'Emoji', 1:'Adet'}, inplace=True)
	return emojidf

def monthly_stats(selected_user, df):
	if selected_user != 'Genel':
		df = df[df['Kullanıcı'] == selected_user]

	temp = df.groupby(['Yıl', 'Ay']).count()['Kullanıcı Mesajı'].reset_index()
	time = []

	for i in range(temp.shape[0]):
		time.append(temp['Ay'][i] + "-" + str(temp['Yıl'][i]))

	temp['Ay-Yıl'] = time
	return temp

def monthly_activity(selected_user, df):
	if selected_user != 'Genel':
		df = df[df['Kullanıcı'] == selected_user]

	return df['Ay'].value_counts()

def weekactivitymap(selected_user, df):
	if selected_user != 'Genel':
		df = df[df['Kullanıcı'] == selected_user]

	return df['Gün Ad'].value_counts()

if __name__ == '__main__':
	st.sidebar.title("Whatsapp Sohbet Analizi")
	uploaded_file = st.sidebar.file_uploader("Sohbet kaydı dosyasını seçin.")

	if uploaded_file is not None:
		bytes_data = uploaded_file.getvalue()
		data = bytes_data.decode("utf-8")
		zaman_turu = ['01.01.2023 12:24']
		pattern = st.sidebar.selectbox("Zaman Türü", zaman_turu)
		df = preprocess(pattern, data)
		user_list = df['Kullanıcı'].unique().tolist()
		user_list.remove('Grup Bildirimi')
		user_list.sort()
		user_list.insert(0, "Genel")
		selected_user = st.sidebar.selectbox("Analiz etmek istediğiniz kişi", user_list)
		if st.sidebar.button("Analizi Göster"):
			st.title(selected_user + " için Sohbet Analizi")
			num_messages, num_words, media_info, links, emojis_len = stats(selected_user, df)

			column1, column2, column3, column4, column5 = st.columns(5)

			with column1:
				st.header("Toplam Mesaj")
				st.title(num_messages)

			with column2:
				st.header("Toplam Kelime")
				st.title(num_words)

			with column3:
				st.header("Medya Sayısı")
				st.title(media_info)

			with column4:
				st.header("Link Sayısı")
				st.title(links)

			with column5:
				st.header("Emoji Sayısı")
				st.title(emojis_len)

			if selected_user == 'Genel':
				st.title("Kullanıcı Sıklığı")

				st.header("Barplot")
				busycount, newdf = user_stats(df)
				fig, ax = plt.subplots()
				ax.bar(busycount.index, busycount.values, color='red')
				plt.xticks(rotation='vertical')
				st.pyplot(fig)

				st.header("PIE Chart")
				fig, ax = plt.subplots()
				pie_y = busycount.values.tolist()
				pie_labels = busycount.index.tolist()
				ax.pie(pie_y, labels=pie_labels, startangle=0, autopct='%1.1f%%')
				st.pyplot(fig)

			if selected_user != 'Genel':
				st.title('Kullanılan Kelime Sıklığı')

				st.header("Barplot")
				user_freq = word_freq(selected_user, df)
				fig, ax = plt.subplots()
				words = [word for word, _ in user_freq]
				counts = [counts for _, counts in user_freq]

				ax.bar(words, counts)
				plt.title(f"{selected_user} KULLANICISININ KULLANDIGI EN SIK 15 KELIME")
				plt.ylabel("Frekans")
				plt.xlabel("Kelimeler")
				st.pyplot(fig)

				st.header("WordCloud")
				fig, ax = plt.subplots()
				dict_top = dict(user_freq)
				wordcloud = WordCloud(width=350, height=350, background_color='white', min_font_size=5).generate_from_frequencies(dict_top)

				plt.figure(figsize=(5,9), facecolor=None)
				ax.imshow(wordcloud)
				ax.axis("off")
				plt.show()
				st.pyplot(fig)


			emoji_df = emoji_stats(selected_user, df)
			if emoji_df.empty == False:
				#emoji_df.columns = ['Emoji', 'Adet']

				st.title("Kullanılan Emojiler")
				emojicount = list(emoji_df['Adet'])
				perlist = [(i / sum(emojicount)) * 100 for i in emojicount]
				emoji_df['Yüzde'] = np.array(perlist)

				fig, ax = plt.subplots()
				pie_y2 = emoji_df['Yüzde'].values.tolist()
				pie_labels2 = [c for c in emoji_df['Emoji'].values if c in emoji.distinct_emoji_list(c)]
				ax.pie(pie_y2, labels=pie_labels2, startangle=0, autopct='%1.1f%%')
				st.pyplot(fig)

			st.title("Aylık Mesaj Sayısı")
			time = monthly_stats(selected_user, df)
			fig, ax = plt.subplots()
			ax.plot(time['Ay-Yıl'], time['Kullanıcı Mesajı'], color='blue')
			plt.xticks(rotation='vertical')
			plt.tight_layout()
			st.pyplot(fig)

			st.title("Aktivite Haritası")

			st.header("Günlük Mesaj Sıklığı")
			busy_day = weekactivitymap(selected_user, df)
			fig, ax = plt.subplots()
			ax.bar(busy_day.index, busy_day.values, color='red')
			plt.xticks(rotation='vertical')
			plt.tight_layout()
			st.pyplot(fig)


			st.header("Aylık Mesaj Sıklığı")
			busy_month = monthly_activity(selected_user, df)
			fig, ax = plt.subplots()
			ax.bar(busy_month.index, busy_month.values, color='green')
			plt.xticks(rotation='vertical')
			plt.tight_layout()
			st.pyplot(fig)