# Language Detection System
import numpy as np
import pickle
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
from django.conf import settings

class LanguageDetector:
    """
    سیستم تشخیص زبان با استفاده از Naive Bayes
    """
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.languages = []
        self.model_path = os.path.join(settings.BASE_DIR, 'detection', 'models')
        self.ensure_model_dir()
    
    def ensure_model_dir(self):
        """اطمینان از وجود پوشه models"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
    
    def preprocess_text(self, text):
        """پیش پردازش متن"""
        # حذف کاراکترهای خاص و اعداد
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        # تبدیل به حروف کوچک
        text = text.lower().strip()
        # حذف فاصله های اضافی
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def prepare_training_data(self):
        """آماده سازی داده های آموزشی برای زبان های مختلف"""
        
        # داده های آموزشی برای زبان های مختلف
        training_data = {
            'english': [
                "Hello how are you today",
                "This is a beautiful day",
                "I love programming and technology",
                "The weather is nice today",
                "Python is a great programming language",
                "Machine learning is fascinating",
                "I enjoy reading books",
                "The sun is shining brightly",
                "Coffee tastes really good",
                "Music makes me happy",
                "Travel broadens the mind",
                "Education is very important",
                "Health is wealth indeed",
                "Time flies so fast",
                "Dreams can come true",
                "She went to the market to buy some fruits",
                "He is studying computer science at university",
                "They are planning a trip to the mountains",
                "My favorite color is blue",
                "The cat is sleeping on the sofa",
                "I have a meeting at 10 o'clock",
                "We watched a movie last night",
                "The children are playing in the park",
                "Please close the window",
                "I need to charge my phone",
                "She likes to paint landscapes",
                "He plays the guitar very well",
                "The train leaves at 7 am",
                "I am learning a new language",
                "Breakfast is the most important meal of the day",
                "The book was very interesting",
                "Can you help me with this problem?",
                "The flowers are blooming in the garden",
                "He is a very talented musician",
                "I forgot my umbrella at home",
                "The restaurant serves delicious food",
                "She is writing a letter to her friend",
                "The dog barked loudly at the stranger",
                "We are going to the beach this weekend",
                "He bought a new car yesterday",
                "The teacher explained the lesson clearly",
                "I enjoy listening to classical music",
                "The city is crowded during rush hour",
                "She wore a beautiful red dress",
                "He is reading the newspaper",
                "The cake tastes sweet and soft",
                "I will call you in the evening",
                "The river flows through the valley",
                "She is afraid of spiders",
                "He likes to play chess with his brother"
            ],
            'persian': [
                "سلام چطور هستید امروز",
                "امروز روز زیبایی است",
                "من عاشق برنامه نویسی هستم",
                "هوا امروز خیلی خوب است",
                "پایتون زبان برنامه نویسی عالی است",
                "یادگیری ماشین جذاب است",
                "من از خواندن کتاب لذت می برم",
                "خورشید به زیبایی می درخشد",
                "قهوه طعم فوق العاده ای دارد",
                "موسیقی مرا شاد می کند",
                "سفر ذهن را گسترش می دهد",
                "آموزش بسیار مهم است",
                "سلامتی گنج است",
                "زمان خیلی سریع می گذرد",
                "رویاها می توانند محقق شوند",
                "او به بازار رفت تا میوه بخرد",
                "او در دانشگاه علوم کامپیوتر می‌خواند",
                "آنها برنامه‌ریزی سفر به کوه دارند",
                "رنگ مورد علاقه من آبی است",
                "گربه روی مبل خوابیده است",
                "من ساعت ۱۰ جلسه دارم",
                "دیشب فیلم تماشا کردیم",
                "بچه‌ها در پارک بازی می‌کنند",
                "لطفاً پنجره را ببند",
                "باید گوشی‌ام را شارژ کنم",
                "او دوست دارد منظره نقاشی کند",
                "او گیتار را خیلی خوب می‌نوازد",
                "قطار ساعت ۷ صبح حرکت می‌کند",
                "در حال یادگیری زبان جدید هستم",
                "صبحانه مهم‌ترین وعده غذایی است",
                "کتاب بسیار جالب بود",
                "می‌توانی در این مسئله به من کمک کنی؟",
                "گل‌ها در باغ شکوفه داده‌اند",
                "او موسیقیدان بسیار با استعدادی است",
                "چترم را در خانه جا گذاشتم",
                "رستوران غذای خوشمزه‌ای سرو می‌کند",
                "او در حال نوشتن نامه به دوستش است",
                "سگ به غریبه با صدای بلند پارس کرد",
                "این آخر هفته به ساحل می‌رویم",
                "او دیروز ماشین جدید خرید",
                "معلم درس را به وضوح توضیح داد",
                "من از گوش دادن به موسیقی کلاسیک لذت می‌برم",
                "شهر در ساعت شلوغی پرجمعیت است",
                "او لباس قرمز زیبایی پوشیده بود",
                "او روزنامه می‌خواند",
                "کیک طعم شیرین و نرمی دارد",
                "عصر با تو تماس می‌گیرم",
                "رودخانه از دره عبور می‌کند",
                "او از عنکبوت می‌ترسد",
                "او دوست دارد با برادرش شطرنج بازی کند"
            ],
            'arabic': [
                "مرحبا كيف حالك اليوم",
                "هذا يوم جميل جداً",
                "أحب البرمجة والتكنولوجيا",
                "الطقس جميل اليوم",
                "بايثون لغة برمجة رائعة",
                "تعلم الآلة مذهل",
                "أستمتع بقراءة الكتب",
                "الشمس تشرق بجمال",
                "القهوة لذيذة جداً",
                "الموسيقى تسعدني كثيراً",
                "السفر يوسع العقل",
                "التعليم مهم جداً",
                "الصحة كنز ثمين",
                "الوقت يمر بسرعة",
                "الأحلام يمكن أن تتحقق",
                "ذهبت إلى السوق لشراء بعض الفواكه",
                "يدرس علوم الحاسوب في الجامعة",
                "يخططون لرحلة إلى الجبال",
                "لوني المفضل هو الأزرق",
                "القطة نائمة على الأريكة",
                "لدي اجتماع في الساعة العاشرة",
                "شاهدنا فيلماً الليلة الماضية",
                "الأطفال يلعبون في الحديقة",
                "من فضلك أغلق النافذة",
                "أحتاج إلى شحن هاتفي",
                "تحب رسم المناظر الطبيعية",
                "يعزف الجيتار بشكل جيد جداً",
                "يغادر القطار الساعة السابعة صباحاً",
                "أتعلم لغة جديدة",
                "الإفطار هو أهم وجبة في اليوم",
                "كان الكتاب ممتعاً جداً",
                "هل يمكنك مساعدتي في هذه المشكلة؟",
                "الأزهار تتفتح في الحديقة",
                "هو موسيقي موهوب جداً",
                "نسيت مظلتي في المنزل",
                "يقدم المطعم طعاماً لذيذاً",
                "تكتب رسالة إلى صديقتها",
                "نبح الكلب بصوت عالٍ على الغريب",
                "سنذهب إلى الشاطئ هذا الأسبوع",
                "اشترى سيارة جديدة أمس",
                "شرح المعلم الدرس بوضوح",
                "أستمتع بالاستماع إلى الموسيقى الكلاسيكية",
                "المدينة مزدحمة خلال ساعة الذروة",
                "ارتدت فستاناً أحمر جميلاً",
                "يقرأ الجريدة",
                "طعم الكعكة حلو وناعم",
                "سأتصل بك في المساء",
                "النهر يجري عبر الوادي",
                "هي تخاف من العناكب",
                "يحب لعب الشطرنج مع أخيه"
            ],
            'french': [
                "Bonjour comment allez vous aujourd'hui",
                "C'est une belle journée",
                "J'aime la programmation et la technologie",
                "Le temps est agréable aujourd'hui",
                "Python est un excellent langage de programmation",
                "L'apprentissage automatique est fascinant",
                "J'aime lire des livres",
                "Le soleil brille magnifiquement",
                "Le café a vraiment bon goût",
                "La musique me rend heureux",
                "Voyager élargit l'esprit",
                "L'éducation est très importante",
                "La santé c'est la richesse",
                "Le temps passe si vite",
                "Les rêves peuvent devenir réalité",
                "Elle est allée au marché acheter des fruits",
                "Il étudie l'informatique à l'université",
                "Ils prévoient un voyage à la montagne",
                "Ma couleur préférée est le bleu",
                "Le chat dort sur le canapé",
                "J'ai une réunion à dix heures",
                "Nous avons regardé un film hier soir",
                "Les enfants jouent dans le parc",
                "Veuillez fermer la fenêtre",
                "Je dois charger mon téléphone",
                "Elle aime peindre des paysages",
                "Il joue très bien de la guitare",
                "Le train part à sept heures du matin",
                "J'apprends une nouvelle langue",
                "Le petit-déjeuner est le repas le plus important de la journée",
                "Le livre était très intéressant",
                "Peux-tu m'aider avec ce problème ?",
                "Les fleurs fleurissent dans le jardin",
                "Il est un musicien très talentueux",
                "J'ai oublié mon parapluie à la maison",
                "Le restaurant sert de la nourriture délicieuse",
                "Elle écrit une lettre à son amie",
                "Le chien a aboyé fort sur l'étranger",
                "Nous allons à la plage ce week-end",
                "Il a acheté une nouvelle voiture hier",
                "Le professeur a expliqué la leçon clairement",
                "J'aime écouter de la musique classique",
                "La ville est bondée pendant l'heure de pointe",
                "Elle portait une belle robe rouge",
                "Il lit le journal",
                "Le gâteau a un goût sucré et doux",
                "Je t'appellerai ce soir",
                "La rivière traverse la vallée",
                "Elle a peur des araignées",
                "Il aime jouer aux échecs avec son frère"
            ],
            'german': [
                "Hallo wie geht es dir heute",
                "Das ist ein schöner Tag",
                "Ich liebe Programmierung und Technologie",
                "Das Wetter ist heute schön",
                "Python ist eine großartige Programmiersprache",
                "Maschinelles Lernen ist faszinierend",
                "Ich lese gerne Bücher",
                "Die Sonne scheint hell",
                "Kaffee schmeckt wirklich gut",
                "Musik macht mich glücklich",
                "Reisen erweitert den Horizont",
                "Bildung ist sehr wichtig",
                "Gesundheit ist Reichtum",
                "Die Zeit vergeht so schnell",
                "Träume können wahr werden",
                "Sie ging zum Markt, um Obst zu kaufen",
                "Er studiert Informatik an der Universität",
                "Sie planen eine Reise in die Berge",
                "Meine Lieblingsfarbe ist Blau",
                "Die Katze schläft auf dem Sofa",
                "Ich habe um zehn Uhr ein Meeting",
                "Wir haben gestern Abend einen Film gesehen",
                "Die Kinder spielen im Park",
                "Bitte schließe das Fenster",
                "Ich muss mein Handy aufladen",
                "Sie malt gerne Landschaften",
                "Er spielt sehr gut Gitarre",
                "Der Zug fährt um sieben Uhr morgens ab",
                "Ich lerne eine neue Sprache",
                "Frühstück ist die wichtigste Mahlzeit des Tages",
                "Das Buch war sehr interessant",
                "Kannst du mir bei diesem Problem helfen?",
                "Die Blumen blühen im Garten",
                "Er ist ein sehr talentierter Musiker",
                "Ich habe meinen Regenschirm zu Hause vergessen",
                "Das Restaurant serviert leckeres Essen",
                "Sie schreibt einen Brief an ihre Freundin",
                "Der Hund bellte laut den Fremden an",
                "Wir gehen dieses Wochenende an den Strand",
                "Er hat gestern ein neues Auto gekauft",
                "Der Lehrer hat die Lektion klar erklärt",
                "Ich höre gerne klassische Musik",
                "Die Stadt ist während der Hauptverkehrszeit überfüllt",
                "Sie trug ein schönes rotes Kleid",
                "Er liest die Zeitung",
                "Der Kuchen schmeckt süß und weich",
                "Ich rufe dich am Abend an",
                "Der Fluss fließt durch das Tal",
                "Sie hat Angst vor Spinnen",
                "Er spielt gerne Schach mit seinem Bruder"
            ],
            'spanish': [
                "Hola como estas hoy",
                "Este es un hermoso día",
                "Me encanta la programación y la tecnología",
                "El clima está agradable hoy",
                "Python es un excelente lenguaje de programación",
                "El aprendizaje automático es fascinante",
                "Disfruto leyendo libros",
                "El sol brilla magníficamente",
                "El café sabe realmente bien",
                "La música me hace feliz",
                "Viajar amplía la mente",
                "La educación es muy importante",
                "La salud es riqueza",
                "El tiempo pasa muy rápido",
                "Los sueños pueden hacerse realidad",
                "Ella fue al mercado a comprar frutas",
                "Él estudia informática en la universidad",
                "Están planeando un viaje a las montañas",
                "Mi color favorito es el azul",
                "El gato está durmiendo en el sofá",
                "Tengo una reunión a las diez",
                "Vimos una película anoche",
                "Los niños juegan en el parque",
                "Por favor, cierra la ventana",
                "Necesito cargar mi teléfono",
                "A ella le gusta pintar paisajes",
                "Él toca muy bien la guitarra",
                "El tren sale a las siete de la mañana",
                "Estoy aprendiendo un nuevo idioma",
                "El desayuno es la comida más importante del día",
                "El libro fue muy interesante",
                "¿Puedes ayudarme con este problema?",
                "Las flores están floreciendo en el jardín",
                "Él es un músico muy talentoso",
                "Olvidé mi paraguas en casa",
                "El restaurante sirve comida deliciosa",
                "Ella está escribiendo una carta a su amiga",
                "El perro ladró fuerte al extraño",
                "Vamos a la playa este fin de semana",
                "Él compró un coche nuevo ayer",
                "El profesor explicó la lección claramente",
                "Me gusta escuchar música clásica",
                "La ciudad está llena durante la hora punta",
                "Ella llevaba un hermoso vestido rojo",
                "Él está leyendo el periódico",
                "El pastel sabe dulce y suave",
                "Te llamaré por la tarde",
                "El río fluye por el valle",
                "Ella tiene miedo de las arañas",
                "Le gusta jugar al ajedrez con su hermano"
            ],
            'italian': [
                "Ciao, come stai oggi?",
                "Questa è una bellissima giornata",
                "Amo programmare e la tecnologia",
                "Il tempo è bello oggi",
                "Python è un ottimo linguaggio di programmazione",
                "L'apprendimento automatico è affascinante",
                "Mi piace leggere libri",
                "Il sole splende luminoso",
                "Il caffè ha un sapore davvero buono",
                "La musica mi rende felice",
                "Viaggiare allarga la mente",
                "L'istruzione è molto importante",
                "La salute è davvero una ricchezza",
                "Il tempo vola così in fretta",
                "I sogni possono diventare realtà",
                "Lei è andata al mercato a comprare della frutta",
                "Lui studia informatica all'università",
                "Stanno pianificando un viaggio in montagna",
                "Il mio colore preferito è il blu",
                "Il gatto dorme sul divano",
                "Ho una riunione alle dieci",
                "Abbiamo guardato un film ieri sera",
                "I bambini giocano nel parco",
                "Per favore, chiudi la finestra",
                "Devo caricare il mio telefono",
                "Le piace dipingere paesaggi",
                "Lui suona molto bene la chitarra",
                "Il treno parte alle sette del mattino",
                "Sto imparando una nuova lingua",
                "La colazione è il pasto più importante della giornata"
            ],
            'russian': [
                "Привет, как дела сегодня?",
                "Сегодня прекрасный день",
                "Я люблю программирование и технологии",
                "Погода сегодня хорошая",
                "Python — отличный язык программирования",
                "Машинное обучение — это увлекательно",
                "Мне нравится читать книги",
                "Солнце ярко светит",
                "Кофе действительно вкусный",
                "Музыка делает меня счастливым",
                "Путешествия расширяют кругозор",
                "Образование очень важно",
                "Здоровье — это богатство",
                "Время летит так быстро",
                "Мечты могут сбываться",
                "Она пошла на рынок за фруктами",
                "Он изучает информатику в университете",
                "Они планируют поездку в горы",
                "Мой любимый цвет — синий",
                "Кот спит на диване",
                "У меня встреча в десять часов",
                "Мы смотрели фильм прошлой ночью",
                "Дети играют в парке",
                "Пожалуйста, закрой окно",
                "Мне нужно зарядить телефон",
                "Ей нравится рисовать пейзажи",
                "Он очень хорошо играет на гитаре",
                "Поезд отправляется в семь утра",
                "Я учу новый язык",
                "Завтрак — самый важный прием пищи за день"
            ],
            'turkish': [
                "Merhaba, bugün nasılsın?",
                "Bugün çok güzel bir gün",
                "Programlamayı ve teknolojiyi seviyorum",
                "Bugün hava güzel",
                "Python harika bir programlama dili",
                "Makine öğrenimi büyüleyici",
                "Kitap okumayı seviyorum",
                "Güneş parlak bir şekilde parlıyor",
                "Kahve gerçekten çok lezzetli",
                "Müzik beni mutlu ediyor",
                "Seyahat etmek zihni genişletir",
                "Eğitim çok önemlidir",
                "Sağlık gerçekten zenginliktir",
                "Zaman çok hızlı geçiyor",
                "Hayaller gerçek olabilir",
                "O, meyve almak için pazara gitti",
                "Üniversitede bilgisayar bilimi okuyor",
                "Dağlara bir gezi planlıyorlar",
                "En sevdiğim renk mavi",
                "Kedi kanepede uyuyor",
                "Saat onda bir toplantım var",
                "Dün gece bir film izledik",
                "Çocuklar parkta oynuyor",
                "Lütfen pencereyi kapat",
                "Telefonumu şarj etmem gerekiyor",
                "Manzara çizmeyi seviyor",
                "Kardeşiyle satranç oynamayı seviyor",
                "Tren sabah yedide kalkıyor",
                "Yeni bir dil öğreniyorum",
                "Kahvaltı günün en önemli öğünüdür"
            ],
            'chinese': [
                "你好，今天怎么样？",
                "今天天气很好",
                "我喜欢编程和技术",
                "今天是美好的一天",
                "Python是一门很棒的编程语言",
                "机器学习很有趣",
                "我喜欢读书",
                "太阳明亮地照耀着",
                "咖啡真的很好喝",
                "音乐让我开心",
                "旅行开阔视野",
                "教育非常重要",
                "健康就是财富",
                "时间过得真快",
                "梦想可以成真",
                "她去市场买水果",
                "他在大学学习计算机科学",
                "他们计划去山里旅行",
                "我最喜欢的颜色是蓝色",
                "猫在沙发上睡觉",
                "我十点有个会议",
                "我们昨晚看了一部电影",
                "孩子们在公园玩耍",
                "请关上窗户",
                "我需要给手机充电",
                "她喜欢画风景画",
                "他吉他弹得很好",
                "火车早上七点出发",
                "我正在学习一门新语言",
                "早餐是一天中最重要的一餐"
            ],
            'japanese': [
                "こんにちは、今日はどうですか？",
                "今日は素晴らしい日です",
                "私はプログラミングと技術が大好きです",
                "今日は天気がいいです",
                "Pythonは素晴らしいプログラミング言語です",
                "機械学習は魅力的です",
                "本を読むのが好きです",
                "太陽が明るく輝いています",
                "コーヒーは本当に美味しいです",
                "音楽は私を幸せにします",
                "旅行は心を広げます",
                "教育はとても重要です",
                "健康は本当に財産です",
                "時間がとても早く過ぎます",
                "夢は叶うことができます",
                "彼女は果物を買いに市場へ行きました",
                "彼は大学でコンピュータサイエンスを勉強しています",
                "彼らは山への旅行を計画しています",
                "私の好きな色は青です",
                "猫はソファで寝ています",
                "10時に会議があります",
                "昨夜映画を見ました",
                "子供たちは公園で遊んでいます",
                "窓を閉めてください",
                "携帯電話を充電する必要があります",
                "彼女は風景画を描くのが好きです",
                "彼はギターをとても上手に弾きます",
                "列車は朝7時に出発します",
                "新しい言語を学んでいます",
                "朝食は一日の中で最も重要な食事です"
            ],
            'hindi': [
                "नमस्ते, आज आप कैसे हैं?",
                "आज बहुत सुंदर दिन है",
                "मुझे प्रोग्रामिंग और तकनीक पसंद है",
                "आज मौसम अच्छा है",
                "Python एक बेहतरीन प्रोग्रामिंग भाषा है",
                "मशीन लर्निंग बहुत रोचक है",
                "मुझे किताबें पढ़ना पसंद है",
                "सूरज तेज़ी से चमक रहा है",
                "कॉफी वाकई में बहुत अच्छी है",
                "संगीत मुझे खुश करता है",
                "यात्रा मन को विस्तृत करती है",
                "शिक्षा बहुत महत्वपूर्ण है",
                "स्वास्थ्य ही धन है",
                "समय बहुत तेज़ी से बीतता है",
                "सपने सच हो सकते हैं",
                "वह फल खरीदने के लिए बाज़ार गई",
                "वह विश्वविद्यालय में कंप्यूटर विज्ञान पढ़ रहा है",
                "वे पहाड़ों की यात्रा की योजना बना रहे हैं",
                "मेरा पसंदीदा रंग नीला है",
                "बिल्ली सोफे पर सो रही है",
                "मेरी दस बजे मीटिंग है",
                "हमने कल रात एक फिल्म देखी",
                "बच्चे पार्क में खेल रहे हैं",
                "कृपया खिड़की बंद करें",
                "मुझे अपना फोन चार्ज करना है",
                "उसे परिदृश्य चित्र बनाना पसंद है",
                "वह बहुत अच्छा गिटार बजाता है",
                "ट्रेन सुबह सात बजे निकलती है",
                "मैं एक नई भाषा सीख रहा हूँ",
                "नाश्ता दिन का सबसे महत्वपूर्ण भोजन है"
            ]
        }
        
        # تبدیل به فرمت مناسب برای آموزش
        texts = []
        labels = []
        
        for language, sentences in training_data.items():
            for sentence in sentences:
                texts.append(self.preprocess_text(sentence))
                labels.append(language)
        
        return np.array(texts), np.array(labels)
    
    def train_model(self, test_size=0.2, random_state=42):
        """آموزش مدل تشخیص زبان"""
        print("در حال آماده سازی داده های آموزشی...")
        texts, labels = self.prepare_training_data()
        
        print("در حال تقسیم داده ها...")
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        print("در حال ایجاد bag of words...")
        self.vectorizer = CountVectorizer(
            max_features=5000,  # حداکثر تعداد کلمات
            ngram_range=(1, 2),  # استفاده از unigrams و bigrams
            min_df=1,  # حداقل تکرار کلمه
            max_df=0.8  # حداکثر تکرار کلمه
        )
        
        X_train_vectors = self.vectorizer.fit_transform(X_train)
        X_test_vectors = self.vectorizer.transform(X_test)
        
        print("در حال آموزش مدل...")
        self.model = MultinomialNB(alpha=1.0)
        self.model.fit(X_train_vectors, y_train)
        
        # ذخیره لیست زبان ها
        self.languages = list(set(labels))
        
        print("در حال ارزیابی مدل...")
        y_pred = self.model.predict(X_test_vectors)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"دقت مدل: {accuracy:.2%}")
        print("\nگزارش تفصیلی:")
        print(classification_report(y_test, y_pred))
        
        # ذخیره مدل
        self.save_model()
        
        return accuracy
    
    def predict_language(self, text):
        """تشخیص زبان متن ورودی"""
        if self.model is None or self.vectorizer is None:
            if not self.load_model():
                raise Exception("مدل آموزش داده نشده است. ابتدا مدل را آموزش دهید.")
        
        # پیش پردازش متن
        processed_text = self.preprocess_text(text)
        
        if not processed_text.strip():
            return None, 0.0
        
        # تبدیل به vector
        text_vector = self.vectorizer.transform([processed_text])
        
        # پیش بینی
        prediction = self.model.predict(text_vector)[0]
        
        # احتمال پیش بینی
        probabilities = self.model.predict_proba(text_vector)[0]
        max_prob = max(probabilities)
        
        return prediction, max_prob
    
    def predict_with_confidence(self, text):
        """تشخیص زبان با جزئیات بیشتر"""
        if self.model is None or self.vectorizer is None:
            if not self.load_model():
                raise Exception("مدل آموزش داده نشده است.")
        
        processed_text = self.preprocess_text(text)
        
        if not processed_text.strip():
            return {
                'predicted_language': None,
                'confidence': 0.0,
                'all_probabilities': {},
                'text_length': 0
            }
        
        text_vector = self.vectorizer.transform([processed_text])
        
        # پیش بینی
        prediction = self.model.predict(text_vector)[0]
        probabilities = self.model.predict_proba(text_vector)[0]
        
        # ایجاد دیکشنری احتمالات
        prob_dict = {}
        for i, lang in enumerate(self.model.classes_):
            prob_dict[lang] = float(probabilities[i])
        
        return {
            'predicted_language': prediction,
            'confidence': float(max(probabilities)),
            'all_probabilities': prob_dict,
            'text_length': len(text),
            'processed_text_length': len(processed_text)
        }
    
    def save_model(self):
        """ذخیره مدل آموزش داده شده"""
        try:
            model_file = os.path.join(self.model_path, 'language_model.pkl')
            vectorizer_file = os.path.join(self.model_path, 'vectorizer.pkl')
            languages_file = os.path.join(self.model_path, 'languages.pkl')
            
            with open(model_file, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(vectorizer_file, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            with open(languages_file, 'wb') as f:
                pickle.dump(self.languages, f)
            
            print(f"مدل با موفقیت در {self.model_path} ذخیره شد")
            return True
        
        except Exception as e:
            print(f"خطا در ذخیره مدل: {e}")
            return False
    
    def load_model(self):
        """بارگذاری مدل ذخیره شده"""
        try:
            model_file = os.path.join(self.model_path, 'language_model.pkl')
            vectorizer_file = os.path.join(self.model_path, 'vectorizer.pkl')
            languages_file = os.path.join(self.model_path, 'languages.pkl')
            
            if not all(os.path.exists(f) for f in [model_file, vectorizer_file, languages_file]):
                return False
            
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(vectorizer_file, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            with open(languages_file, 'rb') as f:
                self.languages = pickle.load(f)
            
            print("مدل با موفقیت بارگذاری شد")
            return True
        
        except Exception as e:
            print(f"خطا در بارگذاری مدل: {e}")
            return False
    
    def get_supported_languages(self):
        """دریافت لیست زبان های پشتیبانی شده"""
        return self.languages if self.languages else []


# تابع کمکی برای تست سریع
def quick_test():
    """تست سریع سیستم"""
    detector = LanguageDetector()
    
    # آموزش مدل
    print("=== شروع آموزش مدل ===")
    accuracy = detector.train_model()
    
    # تست نمونه
    test_texts = [
        "Hello how are you today?",
        "سلام حال شما چطور است؟",
        "مرحبا كيف حالك اليوم؟",
        "Bonjour comment allez-vous?",
        "Hola como estas hoy?",
        "Hallo wie geht es dir?"
    ]
    
    print("\n=== تست مدل ===")
    for text in test_texts:
        result = detector.predict_with_confidence(text)
        print(f"متن: '{text}'")
        print(f"زبان تشخیص داده شده: {result['predicted_language']}")
        print(f"اعتماد: {result['confidence']:.2%}")
        print("-" * 50)

if __name__ == "__main__":
    quick_test()