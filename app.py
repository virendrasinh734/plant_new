import os
import numpy as np
from flask import Flask, request, render_template, jsonify,redirect,url_for,send_file
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import requests
import json
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

e={
    "Doddpathre": {
        "GeneralInfo": "Doddapathre, also known as Mexican Mint or Indian Borage, is a herb with aromatic leaves. It is often used in cooking and traditional medicine in India and other countries. The leaves are succulent and have a strong flavor.",
        "MedicinalInfo": "Doddapathre has several medicinal properties. It is used in traditional medicine for its anti-inflammatory, anti-bacterial, and anti-fungal properties. It is also known for its ability to relieve coughs and colds.",
        "CultivationInfo": "Doddapathre can be grown in pots or in the ground. It requires well-drained soil and partial to full sunlight. It can be propagated from stem cuttings and grows well in tropical and subtropical climates."
    },
    "Drumstick": {
        "GeneralInfo": "Drumstick, scientifically known as Moringa oleifera, is a nutritious and fast-growing tree. It is known for its long, slender, and edible seed pods.",
        "MedicinalInfo": "Drumstick is rich in essential nutrients and has many health benefits. It is used in traditional medicine for its anti-inflammatory and antioxidant properties. It is also a good source of vitamins and minerals.",
        "CultivationInfo": "Drumstick trees are easy to grow from seeds or cuttings. They thrive in well-drained soil and require full sunlight. They are drought-resistant and can be grown in various climates."
    },
    "Ekka": {
        "GeneralInfo": "Ekka, also known as Indian Elm or Holoptelea integrifolia, is a deciduous tree native to the Indian subcontinent. It has serrated leaves and a grayish bark.",
        "MedicinalInfo": "Ekka is used in traditional medicine for its anti-inflammatory and analgesic properties. It is also known for its wood, which is used in making agricultural implements.",
        "CultivationInfo": "Ekka trees can be propagated from seeds or cuttings. They prefer well-drained soil and full sunlight. They are drought-tolerant and grow well in tropical and subtropical regions."
    },
    "Eucalyptus": {
        "GeneralInfo": "Eucalyptus is a genus of tall, evergreen trees and shrubs known for its aromatic leaves. It is native to Australia and widely grown for its timber and essential oil production.",
        "MedicinalInfo": "Eucalyptus oil is used in traditional medicine for its various therapeutic properties, including anti-inflammatory and decongestant effects. The timber from eucalyptus trees is also valuable for construction and furniture making.",
        "CultivationInfo": "Eucalyptus trees prefer well-drained soil and full sunlight. They are drought-tolerant and suitable for regions with a Mediterranean climate."
    },
    "Ganigale": {
        "GeneralInfo": "Ganigale, also known as Delonix regia or Royal Poinciana, is a flowering tree with bright red or orange flowers. It is native to Madagascar but is widely cultivated in tropical regions for its ornamental value.",
        "MedicinalInfo": "Ganigale is not commonly used for medicinal purposes, but its bark and leaves have been used in traditional medicine for their astringent and anti-inflammatory properties.",
        "CultivationInfo": "Ganigale trees require well-drained soil and full sunlight. They are drought-resistant and thrive in tropical and subtropical climates."
    },
    "Ganike": {
        "GeneralInfo": "Ganike, also known as Ficus carica or Common Fig, is a deciduous tree or shrub with lobed leaves and sweet, pear-shaped fruits. It is native to the Middle East and western Asia.",
        "MedicinalInfo": "Ganike fruits are known for their sweet taste and nutritional value. They are rich in fiber, vitamins, and minerals. In traditional medicine, figs have been used for their laxative and anti-inflammatory properties.",
        "CultivationInfo": "Ganike trees are relatively easy to cultivate and can be propagated from cuttings. They prefer well-drained soil and full sunlight. They are suitable for temperate and Mediterranean climates."
    },
    "Gasagase": {
        "GeneralInfo": "Gasagase, also known as Poppy seeds, are small oilseeds obtained from the opium poppy plant. They are commonly used in cooking and baking for their nutty flavor and texture.",
        "MedicinalInfo": "Gasagase seeds are not used for medicinal purposes, but they are a good source of essential nutrients such as magnesium, manganese, and dietary fiber.",
        "CultivationInfo": "Gasagase seeds are typically sown directly in well-drained soil. They require full sunlight and can be grown in various climates."
    },
    "Ginger": {
        "GeneralInfo": "Ginger is a flowering plant known for its rhizome, which is used as a spice and for its medicinal properties. It is native to Southeast Asia.",
        "MedicinalInfo": "Ginger has a long history of use in traditional medicine for its anti-nausea, anti-inflammatory, and antioxidant properties. It is also used as a culinary spice.",
        "CultivationInfo": "Ginger can be cultivated from rhizome cuttings. It thrives in well-drained, loamy soil and partial sunlight. It is suitable for tropical and subtropical regions."
    },
    "Globe Amaranth": {
        "GeneralInfo": "Globe Amaranth, scientifically known as Gomphrena globosa, is an annual flowering plant with round, colorful flower heads. It is native to Central and South America and is grown as an ornamental plant.",
        "MedicinalInfo": "Globe Amaranth is not typically used for medicinal purposes. It is primarily cultivated for its attractive and long-lasting flowers, which are used in floral arrangements and crafts.",
        "CultivationInfo": "Globe Amaranth can be grown from seeds in well-drained soil. It requires full sunlight and is suitable for various climates."
    },
    "Guava": {
        "GeneralInfo": "Guava, also known as Psidium guajava, is a tropical fruit tree with sweet and fragrant fruits. It is native to Central America but is cultivated in many tropical and subtropical regions.",
        "MedicinalInfo": "Guava is rich in vitamin C, dietary fiber, and various antioxidants. It is used in traditional medicine for its potential health benefits, including immune system support and digestive health.",
        "CultivationInfo": "Guava trees are relatively easy to cultivate from seeds or cuttings. They prefer well-drained soil and full sunlight. They are well-suited to tropical and subtropical climates."
    },
    "Henna": {
        "GeneralInfo": "Henna, scientifically known as Lawsonia inermis, is a flowering plant known for its leaves, which are dried and ground to make a natural dye used for coloring hair and creating intricate body art designs. It is native to North Africa, West Asia, and South Asia.",
        "MedicinalInfo": "Henna has been used in traditional medicine for its cooling and soothing properties. It is also believed to have antimicrobial and anti-inflammatory effects. The leaves are sometimes applied topically for skin conditions.",
        "CultivationInfo": "Henna plants require well-drained soil and plenty of sunlight. They can be grown in pots or directly in the ground and are well-suited to arid and semi-arid climates."
    },
    "Hibiscus": {
        "GeneralInfo": "Hibiscus is a genus of flowering plants that includes many species known for their colorful and trumpet-shaped flowers. It is native to warm temperate, subtropical, and tropical regions around the world.",
        "MedicinalInfo": "Hibiscus flowers are used in traditional medicine for their potential health benefits, including lowering blood pressure and reducing cholesterol levels. The plant's leaves are also used in herbal teas. Additionally, some hibiscus species have edible parts.",
        "CultivationInfo": "Hibiscus plants thrive in well-drained soil and full sunlight. They are relatively easy to grow and can be cultivated in various climates, making them popular as ornamental and medicinal plants."
    },
    "Honge": {
        "GeneralInfo": "Honge, also known as Pongamia pinnata or Indian Beech, is a deciduous tree native to India. It is known for its pinnate leaves and small, winged seeds.",
        "MedicinalInfo": "Honge seeds are used in traditional medicine for their potential medicinal properties. They contain a natural insecticide and are used for various purposes, including as a source of biodiesel. Additionally, the plant's bark and leaves have been used for their anti-inflammatory and analgesic effects.",
        "CultivationInfo": "Honge trees can be grown from seeds or cuttings. They prefer well-drained soil and full sunlight. They are well-suited to tropical and subtropical regions and are often used in agroforestry systems."
    },
    "Insulin": {
        "GeneralInfo": "Insulin is a peptide hormone produced by the pancreas that plays a crucial role in regulating blood sugar levels in the body. It is essential for the proper functioning of the human body and is used as a medication to manage diabetes.",
        "MedicinalInfo": "Insulin is not a plant, but a hormone produced in the pancreas. It is used as a medication to treat diabetes, a condition characterized by inadequate insulin production or utilization. Insulin therapy is administered through injections or insulin pumps.",
        "CultivationInfo": "Insulin is not cultivated but is produced in pharmaceutical laboratories. It is an essential medication for individuals with diabetes to help control their blood sugar levels and prevent complications."
    },
    "Jackfruit": {
        "GeneralInfo": "Jackfruit, scientifically known as Artocarpus heterophyllus, is a tropical tree known for its large, spiky fruit, which is one of the largest tree-borne fruits in the world. It is native to South and Southeast Asia.",
        "MedicinalInfo": "Jackfruit is a versatile fruit used in culinary applications. It is rich in vitamins, minerals, and dietary fiber. The seeds of the jackfruit can be boiled or roasted and are also edible. In traditional medicine, various parts of the tree have been used for their potential health benefits.",
        "CultivationInfo": "Jackfruit trees thrive in tropical climates with well-drained soil and full sunlight. They are drought-tolerant and can be cultivated from seeds. The fruit is a popular ingredient in various dishes and is considered a staple in some regions."
    },
    "Jasmine": {
        "GeneralInfo": "Jasmine is a genus of fragrant flowering plants known for their white, yellow, or pink blossoms. It is native to tropical and subtropical regions around the world. Jasmine is widely cultivated for its aromatic flowers.",
        "MedicinalInfo": "Jasmine flowers are used in traditional medicine and aromatherapy for their calming and mood-enhancing properties. Jasmine essential oil is also used for its fragrance and potential benefits. In traditional medicine, jasmine tea is consumed for its potential health effects.",
        "CultivationInfo": "Jasmine plants require well-drained soil and full sunlight. They are popular ornamental plants and are often used for making perfumes and teas. Jasmine is well-suited to a variety of climates."
    },
    "Kamakasturi": {
        "GeneralInfo": "Kamakasturi, also known as Abelmoschus moschatus or Musk Mallow, is a herbaceous plant known for its musk-scented seeds. It is native to India and is cultivated for its aromatic seeds.",
        "MedicinalInfo": "Kamakasturi seeds are used in traditional medicine and perfumery for their fragrance. They are believed to have aphrodisiac properties and are used in various herbal preparations. The seeds are also used as a flavoring agent.",
        "CultivationInfo": "Kamakasturi plants require well-drained soil and full sunlight. They are drought-resistant and are commonly cultivated in India and other tropical regions for their aromatic seeds."
    },
    "Kambajala": {
        "GeneralInfo": "Kambajala, also known as Carissa carandas or Karonda, is a shrub or small tree native to India. It produces small, red or black berries that are edible and sour in taste.",
        "MedicinalInfo": "Kambajala berries are used in traditional medicine for their potential health benefits. They are a good source of vitamin C and have been used for their antimicrobial and digestive properties. Additionally, various parts of the plant are used in traditional remedies.",
        "CultivationInfo": "Kambajala plants are relatively easy to grow and can be propagated from seeds or cuttings. They prefer well-drained soil and full sunlight. They are suitable for tropical and subtropical climates and are commonly grown for their edible berries."
    },
    "Kasambruga": {
        "GeneralInfo": "Kasambruga, also known as Chayote or Sechium edule, is a green, wrinkled fruit that belongs to the gourd family. It is native to Mesoamerica but is widely cultivated in various regions around the world.",
        "MedicinalInfo": "Kasambruga is not commonly used for medicinal purposes, but it is a low-calorie, nutrient-rich food. It is a good source of dietary fiber, vitamins, and minerals. The fruit can be eaten raw or cooked in various dishes.",
        "CultivationInfo": "Kasambruga plants require well-drained soil and full sunlight. They are often grown on trellises to support the climbing vines. Kasambruga is suitable for subtropical and tropical climates and can be propagated from the fruit's seed or vegetative cuttings."
    },
    "Kepala": {
        "GeneralInfo": "Kepala, also known as Watermelon or Citrullus lanatus, is a juicy and refreshing fruit known for its sweet, red or pink flesh and green rind. It is native to Africa but is now cultivated in many parts of the world.",
        "MedicinalInfo": "Kepala is a hydrating fruit that is rich in vitamins and antioxidants. It is often consumed in its natural form or used to make juices and smoothies. Some parts of the watermelon plant have been used in traditional medicine for their potential diuretic and anti-inflammatory properties.",
        "CultivationInfo": "Kepala plants require well-drained soil and full sunlight. They are well-suited to warm and temperate climates and can be grown from seeds. Watermelons are a popular summer fruit."
    },
    "Kohlrabi": {
        "GeneralInfo": "Kohlrabi, scientifically known as Brassica oleracea var. gongylodes, is a vegetable known for its bulbous stem, which can be green, purple, or white. It is a member of the cabbage family and is native to Europe.",
        "MedicinalInfo": "Kohlrabi is a low-calorie vegetable that is a good source of vitamins and dietary fiber. It is often used in salads, soups, and side dishes. Some parts of the kohlrabi plant have been used in traditional medicine for their potential antioxidant and anti-inflammatory properties.",
        "CultivationInfo": "Kohlrabi plants require well-drained soil and full sunlight. They are suitable for temperate climates and can be grown from seeds or transplants. Kohlrabi is a versatile and nutritious addition to various culinary dishes."
    },
    "Lantana": {
        "GeneralInfo": "Lantana, scientifically known as Lantana camara, is a flowering plant known for its clusters of small, brightly colored flowers. It is native to the American tropics but has become an invasive species in various parts of the world.",
        "MedicinalInfo": "Lantana is not commonly used for medicinal purposes, but it is cultivated as an ornamental plant for its attractive flowers. However, some traditional remedies have used Lantana for its potential insect-repellent properties. It is important to note that Lantana can be toxic to livestock and wildlife.",
        "CultivationInfo": "Lantana plants prefer well-drained soil and full sunlight. They are commonly grown as ornamental shrubs and are well-suited to tropical and subtropical climates. Care should be taken to prevent its spread in non-native areas."
    },
    "Lemon": {
        "GeneralInfo": "Lemon, scientifically known as Citrus limon, is a citrus fruit known for its bright yellow color and tangy flavor. It is native to South Asia but is cultivated worldwide for its culinary and medicinal uses.",
        "MedicinalInfo": "Lemons are used in traditional medicine and herbal remedies for their potential health benefits. They are a rich source of vitamin C and antioxidants. Lemon juice is known for its antibacterial and antiviral properties. Additionally, lemons are widely used in cooking and baking.",
        "CultivationInfo": "Lemon trees require well-drained soil and full sunlight. They are commonly grown in tropical and subtropical regions and can be propagated from seeds or cuttings. Lemons are a versatile and popular fruit used in various culinary applications."
    },
    "Lemongrass": {
        "GeneralInfo": "Lemongrass, scientifically known as Cymbopogon citratus, is a fragrant herb known for its lemony flavor and aroma. It is native to tropical regions of Asia and is used in culinary and medicinal applications.",
        "MedicinalInfo": "Lemongrass is used in traditional medicine for its potential health benefits, including digestive relief, anti-inflammatory properties, and stress reduction. It is commonly used to make herbal teas and as a culinary herb in various dishes.",
        "CultivationInfo": "Lemongrass plants require well-drained soil and plenty of sunlight. They are often grown in tropical and subtropical regions and can be propagated from stalk cuttings. Lemongrass is popular in Southeast Asian cuisine and is known for its aromatic properties."
    },
    "Malabar Nut": {
        "GeneralInfo": "Malabar Nut, also known as Justicia adhatoda, is a shrub with lance-shaped leaves and is native to South Asia. It is used in traditional medicine for its potential respiratory and anti-inflammatory properties.",
        "MedicinalInfo": "Malabar Nut leaves are commonly used in traditional remedies and herbal preparations for respiratory issues. They contain alkaloids that may help alleviate coughs and colds. The plant is also valued for its antibacterial properties.",
        "CultivationInfo": "Malabar Nut plants require well-drained soil and partial sunlight. They are commonly cultivated in subtropical and tropical regions and can be propagated from cuttings. Malabar Nut is a significant herb in Ayurvedic medicine."
    },
    "Malabar Spinach": {
        "GeneralInfo": "Malabar Spinach, scientifically known as Basella alba, is a leafy green vegetable with thick, succulent leaves. It is native to tropical Asia but is cultivated in various parts of the world.",
        "MedicinalInfo": "Malabar Spinach is a nutritious vegetable used in culinary dishes, salads, and stir-fries. It is a good source of vitamins, minerals, and dietary fiber. Some parts of the plant have been used in traditional medicine for their potential antioxidant and anti-inflammatory properties.",
        "CultivationInfo": "Malabar Spinach plants require well-drained soil and partial to full sunlight. They are suitable for tropical and subtropical climates and can be propagated from seeds or cuttings. This leafy green is commonly used in Asian and African cuisines."
    },
    "Mango": {
        "GeneralInfo": "Mango, scientifically known as Mangifera indica, is a tropical fruit tree known for its sweet and juicy fruits. It is native to South Asia but is now cultivated in many tropical and subtropical regions.",
        "MedicinalInfo": "Mangoes are a popular fruit known for their delicious flavor and nutritional value. They are rich in vitamins, minerals, and antioxidants. In traditional medicine, various parts of the mango tree have been used for their potential health benefits. Mango leaves are used in herbal teas for their potential antidiabetic properties.",
        "CultivationInfo": "Mango trees require well-drained soil and full sunlight. They are commonly grown in tropical and subtropical climates and can be propagated from seeds. Mangoes are a favorite fruit in many parts of the world and are used in a wide range of culinary dishes."
    },
    "Marigold": {
        "GeneralInfo": "Marigold, scientifically known as Tagetes, is a genus of flowering plants known for their bright and vibrant flowers. They are native to North and South America but are cultivated worldwide as ornamental plants.",
        "MedicinalInfo": "Marigold flowers are not typically used for medicinal purposes, but they have been used in traditional remedies for their potential anti-inflammatory and antiseptic properties. Marigolds are often used in herbal creams and ointments. They are also known for their pest-repelling qualities in gardens.",
        "CultivationInfo": "Marigold plants prefer well-drained soil and full sunlight. They are easy to grow and are commonly used as ornamental flowers in gardens and floral arrangements."
    },
    "Mint": {
        "GeneralInfo": "Mint is a group of aromatic herbs that belong to the Mentha genus. They are known for their refreshing flavor and fragrance. Various species of mint are grown and used in culinary and medicinal applications around the world.",
        "MedicinalInfo": "Mint leaves are widely used for their culinary and medicinal purposes. They are known for their digestive and soothing properties. Mint is often used to make herbal teas, flavor dishes, and relieve digestive discomfort. It is also used in various natural remedies.",
        "CultivationInfo": "Mint plants require well-drained soil and partial to full sunlight. They are easy to grow and are commonly cultivated in gardens or pots. Mint is a popular herb used in a wide range of culinary dishes and beverages."
    },
    "Neem": {
        "GeneralInfo": "Neem, scientifically known as Azadirachta indica, is a tree native to the Indian subcontinent. Neem is known for its bitter-tasting leaves and various parts of the tree are used for their medicinal and agricultural properties.",
        "MedicinalInfo": "Neem leaves, oil, and extracts have a long history of use in traditional medicine. They are known for their anti-inflammatory, antibacterial, and antifungal properties. Neem is used to treat various skin conditions, promote oral health, and has potential benefits for agriculture as a natural pesticide.",
        "CultivationInfo": "Neem trees are typically grown in tropical and subtropical regions. They require well-drained soil and full sunlight. Neem is a valuable tree for its medicinal and agricultural uses."
    },
    "Nelavembu": {
        "GeneralInfo": "Nelavembu, also known as Andrographis paniculata, is an herb native to India and Sri Lanka. It is known for its bitter-tasting leaves and is used in traditional Ayurvedic medicine.",
        "MedicinalInfo": "Nelavembu is used in traditional medicine for its potential immunomodulatory and antipyretic (fever-reducing) properties. It is often used to treat various infections and fevers. Nelavembu is commonly prepared as a herbal decoction.",
        "CultivationInfo": "Nelavembu plants require well-drained soil and partial sunlight. They are often grown as a medicinal herb in Ayurvedic gardens and are suitable for tropical and subtropical climates."
    },
    "Nerale": {
        "GeneralInfo": "Nerale, also known as Emblica officinalis or Indian Gooseberry (Amla), is a fruit-bearing tree native to India. It is known for its sour-tasting, green fruits that are rich in vitamin C and other nutrients.",
        "MedicinalInfo": "Nerale is used in traditional medicine for its potential health benefits. Amla is known for its high vitamin C content and antioxidant properties. It is used in Ayurveda for its rejuvenating effects, and it is often consumed as a dietary supplement or as a component of herbal formulations.",
        "CultivationInfo": "Nerale trees require well-drained soil and full sunlight. They are commonly grown in orchards and are well-suited to tropical and subtropical climates. Amla fruits are used in various culinary and medicinal applications."
    },
    "Nooni": {
        "GeneralInfo": "Nooni, also known as Indian Almond or Terminalia catappa, is a tropical tree known for its almond-shaped fruits. It is native to Southeast Asia and the South Pacific region.",
        "MedicinalInfo": "Nooni seeds are used in traditional medicine for their potential health benefits. They are known for their mild laxative and astringent properties. The leaves of the Nooni tree are also used for their anti-inflammatory and wound-healing effects.",
        "CultivationInfo": "Nooni trees require well-drained soil and full sunlight. They are often cultivated in tropical regions and are grown for their edible seeds and the potential medicinal properties of their leaves and bark."
    },
    "Onion": {
        "GeneralInfo": "Onion, scientifically known as Allium cepa, is a widely cultivated vegetable known for its pungent and layered bulbs. Onions are used in culinary dishes around the world.",
        "MedicinalInfo": "Onions are not typically used for medicinal purposes, but they have been used in traditional remedies for their potential anti-inflammatory and antimicrobial properties. Onions are a versatile cooking ingredient and are used in various savory dishes, sauces, and salads.",
        "CultivationInfo": "Onion plants require well-drained soil and full sunlight. They are commonly grown in gardens and vegetable plots and are suitable for various climates. Onions are a staple in many culinary cuisines."
    },
    "Padri": {
        "GeneralInfo": "Padri, also known as Centella asiatica or Gotu Kola, is a herbaceous plant native to Asia. It is known for its round, green leaves and is used in traditional Ayurvedic and herbal medicine.",
        "MedicinalInfo": "Padri is used in traditional medicine for its potential cognitive and skin-related benefits. It is believed to support mental clarity and skin health. Gotu Kola is often consumed as an herbal tea or used in topical creams and ointments.",
        "CultivationInfo": "Padri plants require well-drained soil and partial to full sunlight. They are often grown as medicinal herbs and are suitable for tropical and subtropical climates."
    },
    "Palak (Spinach)": {
        "GeneralInfo": "Palak, also known as Spinacia oleracea, is a leafy green vegetable known for its dark green, nutritious leaves. It is native to central and western Asia and is widely cultivated for its culinary and nutritional value.",
        "MedicinalInfo": "Palak leaves are used in culinary dishes, salads, and smoothies. They are rich in vitamins, minerals, and dietary fiber. Spinach is a popular choice for salads and as a side dish in various cuisines. While not traditionally used for medicinal purposes, it is valued for its nutritional content.",
        "CultivationInfo": "Palak plants require well-drained soil and partial to full sunlight. They are commonly grown in vegetable gardens and pots. Spinach is a versatile leafy green and a staple in many diets."
    },
    "Papaya": {
        "GeneralInfo": "Papaya, scientifically known as Carica papaya, is a tropical fruit tree known for its large, pear-shaped fruits with orange flesh and black seeds. It is native to Central America but is cultivated in tropical and subtropical regions around the world.",
        "MedicinalInfo": "Papaya fruit is rich in vitamins, minerals, and dietary fiber. It contains an enzyme called papain, which is used in traditional medicine and digestive supplements. Papaya is known for its potential digestive and anti-inflammatory properties. The leaves and seeds are also used in herbal remedies.",
        "CultivationInfo": "Papaya trees require well-drained soil and full sunlight. They are often grown in tropical and subtropical regions. Papaya is commonly propagated from seeds, and the fruit is used in various culinary dishes and beverages."
    },
    "Parijatha": {
        "GeneralInfo": "Parijatha, also known as Nyctanthes arbor-tristis or Night Jasmine, is a fragrant, small tree or shrub native to South Asia. It is known for its white, aromatic flowers that bloom at night.",
        "MedicinalInfo": "Parijatha is used in traditional medicine for its potential health benefits. The leaves and flowers are believed to have anti-inflammatory and antioxidant properties. Parijatha flowers are used to make herbal teas and oils. They are also considered sacred in some cultures.",
        "CultivationInfo": "Parijatha plants require well-drained soil and full sunlight. They are often grown in gardens and temple premises. Parijatha flowers are known for their strong fragrance and are used in various rituals and religious ceremonies."
    },
    "Pea": {
        "GeneralInfo": "Pea, scientifically known as Pisum sativum, is a cool-season vegetable known for its edible green or yellow pods with round seeds. Peas are widely cultivated for culinary purposes around the world.",
        "MedicinalInfo": "Peas are a good source of vitamins, minerals, and dietary fiber. They are used in various culinary dishes, such as soups, salads, and stir-fries. While peas are not commonly used for medicinal purposes, they provide important nutrients in the diet.",
        "CultivationInfo": "Pea plants require well-drained soil and partial sunlight. They are often grown in cool and temperate climates and can be sown directly in the garden. Peas are a versatile and nutritious vegetable."
    },
    "Pepper": {
        "GeneralInfo": "Pepper, also known as Piper nigrum, is a flowering vine known for its spicy and pungent berries, which are used as a spice. Pepper is native to South Asia and is widely used in culinary applications.",
        "MedicinalInfo": "Pepper is not commonly used for medicinal purposes, but it is a popular spice known for its flavor and potential digestive benefits. Black pepper, in particular, is believed to stimulate digestion and has antioxidant properties. It is used as a seasoning in various dishes.",
        "CultivationInfo": "Pepper vines require well-drained soil and partial sunlight. They are commonly cultivated in tropical and subtropical regions. Pepper berries are harvested, dried, and ground to make the spice used in a wide range of cuisines."
    },
    "Pomegranate": {
        "GeneralInfo": "Pomegranate, scientifically known as Punica granatum, is a fruit-bearing shrub or small tree known for its round, red fruits filled with juicy arils (seeds). It is native to the Middle East but is cultivated in various regions.",
        "MedicinalInfo": "Pomegranate is known for its juicy arils, which are a good source of vitamins, antioxidants, and dietary fiber. The fruit is used in culinary dishes, juices, and desserts. In traditional medicine, pomegranate has been used for its potential health benefits, including cardiovascular support and antioxidant effects.",
        "CultivationInfo": "Pomegranate shrubs or trees require well-drained soil and full sunlight. They are often grown in Mediterranean and subtropical regions. Pomegranates are valued for their delicious and nutritious fruit."
    },
    "Pumpkin": {
        "GeneralInfo": "Pumpkin, scientifically known as Cucurbita pepo, is a vine-like vegetable plant known for its round, orange fruits. Pumpkins are native to North America and are widely cultivated for culinary purposes.",
        "MedicinalInfo": "Pumpkin is a nutritious vegetable known for its vibrant orange flesh, which is rich in vitamins, minerals, and dietary fiber. It is used in various culinary dishes, such as soups, pies, and roasted dishes. While not commonly used for medicinal purposes, pumpkin seeds are believed to have potential health benefits, including support for prostate health.",
        "CultivationInfo": "Pumpkin plants require well-drained soil and full sunlight. They are often grown in gardens and vegetable plots. Pumpkins are a versatile ingredient in a variety of dishes and desserts."
    },
    "Radish": {
        "GeneralInfo": "Radish, scientifically known as Raphanus sativus, is a fast-growing root vegetable known for its edible, crisp, and spicy-flavored roots. Radishes are cultivated and consumed in many parts of the world.",
        "MedicinalInfo": "Radishes are a low-calorie vegetable that is a good source of vitamins, minerals, and dietary fiber. They are often used in salads, sandwiches, and as a crunchy snack. In traditional medicine, radishes are believed to have potential digestive and diuretic properties. Additionally, some parts of the plant have been used in herbal remedies.",
        "CultivationInfo": "Radish plants require well-drained soil and full sunlight. They are often grown in gardens and are suitable for cool and temperate climates. Radishes are valued for their fast growth and crisp texture."
    },
    "Rose": {
        "GeneralInfo": "Rose, scientifically known as Rosa, is a genus of flowering plants known for their fragrant and colorful flowers. Roses are native to various regions around the world and are widely cultivated for their ornamental and perfumery value.",
        "MedicinalInfo": "Roses are not commonly used for medicinal purposes, but they are cherished for their sweet fragrance and are used in the production of essential oils and perfumes. Some species of roses are used in traditional medicine for their potential anti-inflammatory and antioxidant properties. Rosewater, a byproduct of rose oil extraction, is used in various cosmetic and culinary applications.",
        "CultivationInfo": "Roses require well-drained soil and full sunlight. They are often grown in gardens and as ornamental plants. Roses are a symbol of love and beauty and are used in various cultural and religious ceremonies."
    },
    "Sampige": {
        "GeneralInfo": "Sampige, also known as Champaca or Magnolia champaca, is a fragrant flowering tree native to Southeast Asia. It is known for its aromatic, yellow or orange flowers that are used in perfumery and religious rituals.",
        "MedicinalInfo": "Sampige flowers are not commonly used for medicinal purposes but are cherished for their sweet fragrance. The flowers are used to make perfumes, garlands, and traditional incense. They are also considered sacred in some cultures and are used in religious ceremonies.",
        "CultivationInfo": "Sampige trees require well-drained soil and full sunlight. They are often grown in gardens, temple premises, and as ornamental trees. Sampige flowers are known for their strong and pleasant fragrance."
    },
    "Sapota": {
        "GeneralInfo": "Sapota, also known as Manilkara zapota or Chikoo, is a tropical fruit tree known for its sweet and grainy-textured fruits. It is native to Central America but is cultivated in tropical regions worldwide.",
        "MedicinalInfo": "Sapota fruit is known for its sweet and flavorful taste. It is a good source of vitamins, minerals, and dietary fiber. While not commonly used for medicinal purposes, it is a delicious and nutritious fruit enjoyed fresh or used in desserts, smoothies, and shakes.",
        "CultivationInfo": "Sapota trees require well-drained soil and full sunlight. They are often grown in tropical and subtropical regions and can be propagated from seeds or grafting. Sapota is a popular fruit in many parts of the world."
    },
    "Seethaashoka": {
        "GeneralInfo": "Seethaashoka, also known as Saraca indica or Ashoka tree, is a small to medium-sized evergreen tree native to India. It is known for its fragrant, bright orange and yellow flowers, which are used in traditional Ayurvedic medicine.",
        "MedicinalInfo": "Seethaashoka flowers are used in traditional medicine for their potential benefits related to female reproductive health. They are believed to support hormonal balance and alleviate various menstrual discomforts. The tree also holds cultural and religious significance.",
        "CultivationInfo": "Seethaashoka trees require well-drained soil and partial sunlight. They are often grown in gardens, temple premises, and as ornamental trees. Seethaashoka flowers are highly valued for their role in Ayurvedic remedies."
    },
    "Seethapala": {
        "GeneralInfo": "Seethapala, also known as Annona squamosa or Sugar-apple, is a fruit-bearing tree native to the American tropics. It is known for its green, scaly fruits with sweet and custard-like pulp.",
        "MedicinalInfo": "Seethapala fruit is known for its sweet and creamy texture. It is a good source of vitamins, minerals, and dietary fiber. The fruit is often consumed fresh or used in desserts and smoothies. While not commonly used for medicinal purposes, some parts of the tree have been used in traditional remedies.",
        "CultivationInfo": "Seethapala trees require well-drained soil and full sunlight. They are often grown in tropical and subtropical regions and can be propagated from seeds or grafting. Seethapala is a popular fruit in many countries."
    },
    "Spinach": {
        "GeneralInfo": "Spinach, scientifically known as Spinacia oleracea, is a leafy green vegetable known for its dark green, nutritious leaves. It is native to central and western Asia and is widely cultivated for culinary and nutritional purposes.",
        "MedicinalInfo": "Spinach leaves are a good source of vitamins, minerals, and dietary fiber. They are used in various culinary dishes, such as salads, soups, and stir-fries. While not commonly used for medicinal purposes, spinach provides important nutrients in the diet.",
        "CultivationInfo": "Spinach plants require well-drained soil and partial to full sunlight. They are often grown in gardens and vegetable plots. Spinach is a versatile and nutritious leafy green vegetable."
    },
    "Tamarind": {
        "GeneralInfo": "Tamarind, scientifically known as Tamarindus indica, is a tropical fruit-bearing tree known for its brown, pod-like fruits with a sweet and tangy pulp. It is native to tropical Africa but is cultivated in many parts of the world.",
        "MedicinalInfo": "Tamarind pulp is used in culinary dishes, chutneys, and beverages. It is known for its sour flavor and is a good source of vitamins and minerals. In traditional medicine, tamarind has been used for its potential digestive and anti-inflammatory properties. The fruit is also used for its tartness and flavor in various cuisines.",
        "CultivationInfo": "Tamarind trees require well-drained soil and full sunlight. They are commonly grown in tropical and subtropical regions and can be propagated from seeds or grafting. Tamarind is valued for its culinary and medicinal uses."
    },
    "Taro": {
        "GeneralInfo": "Taro, scientifically known as Colocasia esculenta, is a tropical root vegetable known for its starchy corms. It is native to Southeast Asia and the Pacific and is widely cultivated for its edible roots and leaves.",
        "MedicinalInfo": "Taro corms and leaves are used in culinary dishes and as a staple food in some regions. The corms are rich in carbohydrates and dietary fiber. While not commonly used for medicinal purposes, taro provides an important source of calories in certain diets.",
        "CultivationInfo": "Taro plants require well-drained soil and partial sunlight. They are often grown in tropical and subtropical regions and can be propagated from corms or cormels. Taro is a versatile and staple food in many cuisines."
    },
    "Tecoma": {
        "GeneralInfo": "Tecoma, also known as Tecoma stans or Yellow Trumpetbush, is a flowering shrub or small tree native to the Americas. It is known for its bright yellow trumpet-shaped flowers.",
        "MedicinalInfo": "Tecoma flowers are not commonly used for medicinal purposes but are valued for their ornamental beauty. The flowers are used to attract pollinators and enhance the aesthetics of gardens and landscapes. Tecoma is often grown as an ornamental plant.",
        "CultivationInfo": "Tecoma plants require well-drained soil and full sunlight. They are commonly cultivated in gardens and landscapes, and they are well-suited to warm and subtropical climates. Tecoma is admired for its vibrant and showy flowers."
    },
    "Thumbe": {
        "GeneralInfo": "Thumbe, also known as Leucas aspera, is an herbaceous plant native to India."
    },
    "Tulsi": {
        "GeneralInfo": "Tulsi, also known as Holy Basil or Ocimum sanctum, is a sacred and aromatic herb native to India. It is known for its fragrant leaves and religious significance in Hindu culture.",
        "MedicinalInfo": "Tulsi is widely used in traditional Ayurvedic medicine for its potential health benefits. It is believed to have anti-inflammatory, antimicrobial, and adaptogenic properties. Tulsi leaves are used to make herbal teas, extracts, and are often consumed fresh. The plant holds cultural and religious significance in India.",
        "CultivationInfo": "Tulsi plants require well-drained soil and partial sunlight. They are often grown in home gardens, temple premises, and as ornamental plants. Tulsi is considered a sacred plant and is an integral part of various rituals and ceremonies."
    },
    "Turmeric": {
        "GeneralInfo": "Turmeric, scientifically known as Curcuma longa, is a flowering plant native to South Asia. It is known for its bright orange-yellow rhizomes, which are used to produce the popular spice, turmeric. Turmeric is widely cultivated for its culinary, medicinal, and coloring properties.",
        "MedicinalInfo": "Turmeric contains a compound called curcumin, which has been extensively studied for its potential health benefits. It is known for its anti-inflammatory and antioxidant properties. Turmeric is used in traditional medicine for various purposes, including pain relief and as an immune system booster. It is a key ingredient in many Indian dishes and is used in traditional remedies.",
        "CultivationInfo": "Turmeric plants require well-drained soil and partial sunlight. They are commonly grown in tropical and subtropical regions. Turmeric rhizomes are harvested, dried, and ground to produce the spice used in a wide range of culinary dishes and beverages."
    },
    "Sandalwood": {
        "GeneralInfo": "A semi-parasitic tree native to India and Southeast Asia. It is known for its fragrant wood, which is used in perfumes, incense, and religious ceremonies.",
        "MedicinalInfo": "Sandalwood oil has been used for centuries in traditional medicine to treat a variety of ailments, including skin diseases, respiratory problems, and digestive disorders.",
        "CultivationInfo": "Sandalwood trees are slow-growing and require a warm, humid climate. They are typically propagated from seed and can take up to 20 years to reach maturity."
    },
    "Jamaica cherry": {
        "GeneralInfo": "A small tree or shrub native to Mexico and Central America. It is known for its sweet, edible fruits, which are used in pies, jams, and other desserts.",
        "MedicinalInfo": "Jamaica cherry leaves have been used in traditional medicine to treat a variety of ailments, including diabetes, diarrhea, and inflammation.",
        "CultivationInfo": "Jamaica cherry trees are easy to grow and can be grown in a variety of climates. They are typically propagated from seed or cuttings."
    },
    "Indian Mint, Mexican mint": {
        "GeneralInfo": "A perennial herb native to Southeast Asia. It is known for its aromatic leaves, which are used in cooking and traditional medicine.",
        "MedicinalInfo": "Indian mint leaves have been used in traditional medicine to treat a variety of ailments, including respiratory problems, digestive disorders, and skin diseases.",
        "CultivationInfo": "Indian mint is a very easy-to-grow herb. It can be grown in the ground or in containers in a variety of climates. It can be propagated from seed, cuttings, or division."
    },
    "Oriental mustard": {
        "GeneralInfo": "A leafy green vegetable native to Asia. It is known for its pungent flavor and is used in a variety of dishes, including salads, stir-fries, and soups.",
        "MedicinalInfo": "Oriental mustard seeds have been used in traditional medicine to treat a variety of ailments, including respiratory problems, digestive disorders, and skin diseases.",
        "CultivationInfo": "Oriental mustard is a cool-season crop that can be grown in a variety of climates. It is typically sown directly in the ground, but it can also be started indoors and transplanted."
    },
    "Aloe vera": {
        "GeneralInfo": "A succulent plant native to Africa and the Middle East. It is known for its thick, fleshy leaves, which contain a gel that has been used for centuries to treat a variety of ailments.",
        "MedicinalInfo": "Aloe vera gel has been used in traditional medicine to treat burns, cuts, scrapes, and other skin conditions. It has also been used to treat digestive disorders, respiratory problems, and inflammatory conditions.",
        "CultivationInfo": "Aloe vera is a very easy-to-grow plant. It can be grown in the ground or in containers in a variety of climates. It can be propagated from offsets or from cuttings."
    },
    "Garlic": {
        "GeneralInfo": "A bulbous plant native to Central Asia. It is known for its pungent flavor and is used in a variety of dishes around the world.",
        "MedicinalInfo": "Garlic has been used in traditional medicine for centuries to treat a variety of ailments, including respiratory problems, cardiovascular disease, and cancer.",
        "CultivationInfo": "Garlic is a cool-season crop that can be grown in a variety of climates. It is typically planted in the fall and harvested in the spring. Garlic can be propagated from cloves or from seed."
    },
    "Purple coneflower": {
        "GeneralInfo": "A flowering plant native to North America. It is known for its bright purple flowers and its immune-boosting properties.",
        "MedicinalInfo": "Purple coneflower has been used in traditional medicine for centuries to treat colds, flu, and other infections.",
        "CultivationInfo": "Purple coneflower is a hardy plant that can be grown in a variety of climates. It is typically propagated from seed or by division."
    },
    "Chamomile": {
        "GeneralInfo": "A flowering plant native to Europe and Asia. It is known for its calming"
    },
    "Aloevera": {
        "GeneralInfo": "A succulent plant with thick, fleshy leaves that contain a gel that is used in a variety of products, including cosmetics, lotions, and sunscreens.",
        "MedicinalInfo": "Aloe vera gel is used to treat a variety of skin conditions, including burns, sunburns, and eczema. It is also used to treat digestive problems, such as constipation and diarrhea.",
        "CultivationInfo": "Aloe vera is a relatively easy plant to grow. It can be grown indoors or outdoors, but it prefers full sun and well-drained soil."
    },
    "Amla": {
        "GeneralInfo": "A deciduous tree native to India and Southeast Asia. The fruit of the amla tree is a small, green fruit that is known for its high vitamin C content.",
        "MedicinalInfo": "Amla fruit is used to treat a variety of health conditions, including anemia, indigestion, and constipation. It is also used to boost the immune system and promote healthy skin.",
        "CultivationInfo": "Amla trees can be grown in a variety of climates, but they prefer warm, humid conditions. They can be grown from seed or cuttings."
    },
    "Amruthaballi": {
        "GeneralInfo": "A deciduous climbing vine native to India and Southeast Asia. The leaves of the amruthaballi plant are used in a variety of Ayurvedic medicines.",
        "MedicinalInfo": "Amruthaballi leaves are used to treat a variety of health conditions, including fever, malaria, and diabetes. They are also used to boost the immune system and promote overall health.",
        "CultivationInfo": "Amruthaballi vines can be grown in a variety of climates, but they prefer warm, humid conditions. They can be grown from seed or cuttings."
    },
    "Arali": {
        "GeneralInfo": "An evergreen tree native to Taiwan and China. The leaves of the arali plant are used in a variety of traditional Chinese medicines.",
        "MedicinalInfo": "Arali leaves are used to treat a variety of health conditions, including arthritis, rheumatism, and pain. They are also used to boost the immune system and promote overall health.",
        "CultivationInfo": "Arali trees can be grown in a variety of climates, but they prefer warm, humid conditions. They can be grown from seed or cuttings."
    },
    "Ashoka": {
        "GeneralInfo": "A deciduous tree native to India and Southeast Asia. The bark of the ashoka tree is used in a variety of Ayurvedic medicines.",
        "MedicinalInfo": "Ashoka bark is used to treat a variety of gynecological problems, including menstrual cramps, heavy bleeding, and infertility. It is also used to treat skin conditions and boost the immune system.",
        "CultivationInfo": "Ashoka trees can be grown in a variety of climates, but they prefer warm, humid conditions. They can be grown from seed or cuttings."
    },
    "Asthma_weed": {
        "GeneralInfo": "A herbaceous annual plant native to tropical and subtropical regions around the world. The whole plant of the asthma weed is used in a variety of traditional medicines.",
        "MedicinalInfo": "Asthma weed is used to treat a variety of respiratory problems, including asthma, bronchitis, and cough. It is also used to treat skin conditions, diarrhea, and fever.",
        "CultivationInfo": "Asthma weed can be grown in a variety of climates, but it prefers warm, humid conditions. It can be grown from seed or cuttings."
    },
    "Badipala": {
        "GeneralInfo": "A deciduous shrub or small tree native to India and Southeast Asia. The leaves of the badipala plant are used in a variety of Ayurvedic medicines.",
        "MedicinalInfo": "Badipala leaves are used to treat a variety of skin conditions, including acne, eczema, and psoriasis. They are also used to treat respiratory problems, such as asthma and bronchitis.",
        "CultivationInfo": "Badipala plants can be grown in a variety of climates, but they prefer warm, humid conditions. They can be grown from seed or cuttings."
    },
    "Balloon_Vine": {
        "GeneralInfo": "A herbaceous annual vine native to tropical and subtropical regions around the world. The whole plant of the balloon vine is used in a variety of traditional medicines.",
        "MedicinalInfo": "Balloon vine is used to treat a variety of health conditions, including kidney stones, urinary tract infections, and skin conditions. It is also used to boost the immune system and promote overall health.",
        "CultivationInfo": "Balloon vines can be grown in a variety of climates, but they prefer warm, humid conditions. They can be grown from seed or cuttings."
    },
    "Bamboo": {
        "GeneralInfo": "A subfamily of grasses that is native to tropical and subtropical regions around the world. The stems of bamboo plants are used in a variety of products, including furniture, construction materials, and musical instruments.",
        "MedicinalInfo": "Bamboo shoots are used in a variety of Asian cuisines. They are a good source of fiber and vitamins A and C.",
        "CultivationInfo": "Bamboo plants can be grown in a variety of climates, but they prefer warm, humid conditions. They can be grown from seed."
    },
    "Betel": {
        "GeneralInfo": "A perennial evergreen climbing vine native to Southeast Asia. The leaves of the betel vine are used in a variety of Asian cultures.",
        "MedicinalInfo": "Betel leaves are used to freshen breath and improve digestion. They are also used in traditional medicine to treat a variety of health conditions, such as asthma, bronchitis, and diarrhea.",
        "CultivationInfo": "Betel vines can be grown in a variety of climates, but they prefer warm, humid conditions. They can be grown from seed or cuttings."
    },
    "Bringaraja": {
        "GeneralInfo": "A perennial herbaceous plant native to tropical and subtropical regions around the world. The whole plant of the bringaraja is used in a variety of traditional medicines.",
        "MedicinalInfo": "Bringaraja is used to treat a variety of health conditions, including liver problems, hair loss, and skin conditions. It is also used to boost the immune system and promote overall health.",
        "CultivationInfo": "Bringaraja plants can be grown in a variety of climates, but they prefer warm, humid conditions. They can be grown from seed or cuttings."
    },
    "Beans": {
        "GeneralInfo": "A herbaceous annual plant native to Central and South America. The seeds of the bean plant are a staple food in many cultures.",
        "MedicinalInfo": "Beans are a good source of protein, fiber, and vitamins. They are also low in fat and calories.",
        "CultivationInfo": "Beans can be grown in a variety of climates, but they prefer warm, sunny conditions. They can be grown from seed or seedlings."
    },
    "Bhrami": {
        "GeneralInfo": "A perennial herbaceous plant native to India and Southeast Asia. The leaves of the bhrami plant are used in a variety of Ayurvedic medicines.",
        "MedicinalInfo": "Bhrami leaves are used to improve memory, cognitive function, and focus. They are also used to treat anxiety, depression, and insomnia.",
        "CultivationInfo": "Bhrami plants can be grown in a variety of climates, but they prefer warm, humid conditions. They can be grown from seed or cuttings."
    },
    "Camphor": {
        "GeneralInfo": "Camphor is a waxy, translucent solid with a strong, characteristic aroma. It is obtained from the bark and leaves of the camphor laurel tree. Camphor is used in a variety of products, including cosmetics, toiletries, and pharmaceuticals.",
        "MedicinalInfo": "Camphor has a number of medicinal properties, including anti-inflammatory, analgesic, and antiseptic properties. It is used to treat a variety of conditions, including muscle aches, headaches, and respiratory infections.",
        "CultivationInfo": "Camphor laurel trees are native to Asia and Africa. They can grow up to 30 meters tall and have large, glossy leaves. Camphor laurel trees are grown in tropical and subtropical climates."
    },
    "Caricature": {
        "GeneralInfo": "A caricature is a portrait of a person or thing that is exaggerated or distorted in order to highlight its humorous or grotesque aspects. Caricatures are often used in cartoons, comic strips, and political satire.",
        "MedicinalInfo": "Caricatures have no medicinal information.",
        "CultivationInfo": "Caricatures are not cultivated."
    },
    "Castor": {
        "GeneralInfo": "Castor is a plant that is native to India and Africa. It is grown for its seeds, which contain a valuable oil called castor oil. Castor oil is used in a variety of products, including lubricants, cosmetics, and pharmaceuticals.",
        "MedicinalInfo": "Castor oil has a number of medicinal properties, including laxative, anti-inflammatory, and antibacterial properties. It is used to treat a variety of conditions, including constipation, skin conditions, and infections.",
        "CultivationInfo": "Castor plants are grown in tropical and subtropical climates. They are easy to grow and can be grown in a variety of soils."
    },
    "Catharanthus": {
        "GeneralInfo": "Catharanthus is a genus of flowering plants in the dogbane family. It is native to Madagascar and Africa. Catharanthus is grown for its alkaloids, which are used in a variety of medications, including cancer drugs and vincristine.",
        "MedicinalInfo": "Catharanthus alkaloids have a number of medicinal properties, including anti-cancer, anti-malarial, and anti-diabetic properties. They are used to treat a variety of conditions, including cancer, malaria, and diabetes.",
        "CultivationInfo": "Catharanthus plants are grown in tropical and subtropical climates. They are easy to grow and can be grown in a variety of soils."
    },
    "Chakte": {
        "GeneralInfo": "Chakte is a type of chili pepper that is native to Mexico. It is known for its smoky flavor and its medium heat. Chakte peppers are used in a variety of Mexican dishes, including mole and adobo sauce.",
        "MedicinalInfo": "Chakte peppers have a number of medicinal properties, including anti-inflammatory, analgesic, and antioxidant properties. They are used to treat a variety of conditions, including muscle aches, headaches, and digestive problems.",
        "CultivationInfo": "Chakte pepper plants are grown in tropical and subtropical climates. They are easy to grow and can be grown in a variety of soils."
    },
    "Chilly": {
        "GeneralInfo": "A chili pepper is a type of fruit that is native to Central and South America. Chili peppers are known for their spicy flavor and their high vitamin C content. Chili peppers are used in a variety of cuisines around the world.",
        "MedicinalInfo": "Chili peppers have a number of medicinal properties, including anti-inflammatory, analgesic, and anti-cancer properties. They are used to treat a variety of conditions, including muscle aches, headaches, and cancer.",
        "CultivationInfo": "Chili pepper plants are grown in tropical and subtropical climates. They are easy to grow and can be grown in a variety of soils."
    },
    "Citron lime": {
        "GeneralInfo": "Citron lime is a type of citrus fruit that is native to India. It is known for its sour taste and its high vitamin C content. Citron limes are used in a variety of Indian dishes, including pickles and chutneys.",
        "MedicinalInfo": "Citron limes have a number of medicinal properties, including anti-inflammatory, analgesic, and antioxidant properties. They are used to treat a variety of conditions, including muscle aches, headaches, and digestive problems.",
        "CultivationInfo": "Citron lime trees are grown in tropical and subtropical climates. They are easy to grow and can be grown in a variety of soils."
    },
    "Coffee": {
        "GeneralInfo": "Coffee is a beverage that is made from the roasted seeds of the coffee plant. Coffee is known for its bitter taste and its stimulating effects. Coffee is consumed around the world and is one of the most popular beverages.",
        "MedicinalInfo": "Coffee has a number of medicinal properties, including anti-inflammatory, antioxidant, and anti-cancer properties. Coffee is used to treat a variety of conditions, including type 2 diabetes, Alzheimer's disease, and Parkinson's disease.",
        "CultivationInfo": "Coffee plants are grown in tropical and subtropical climates. They are easy to grow and can be grown in a variety of soils."
    },
    "Common rue": {
        "GeneralInfo": "Common rue is a type of herb that is native to Europe and Asia. It is known for its strong, acrid flavor. Common rue is used in a variety of cuisines around the world and is also used in traditional medicine.",
        "MedicinalInfo": "Common rue has a number of medicinal properties, including anti-inflammatory, analgesic, and antioxidant properties.",
        "CultivationInfo": "Common rue plants are grown in various regions."
    }
}
model = tf.keras.models.load_model('my_mmodel.h5')
with open("labels.txt","r") as file:
    s=file.readlines()
r=[s[i].strip() for i in range(len(s)) ]
def predict_image(file_path):
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.sigmoid(predictions[0])
    print("This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(r[np.argmax(score)], 100 * np.max(score)))
    return r[np.argmax(score)]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file extension'})
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process the uploaded image and make predictions
        prediction = [predict_image(file_path)]
        info="this "
        info=e[prediction[0]]
        print("info is",info)
        return render_template('ret.html', data=[prediction,info])

    return render_template('index.html')
@app.route('/try', methods=['GET', 'POST'])
def tryh():
    return render_template("try.html")


@app.route('/ret')
def ret():
    message = request.args.get('message', '')
    section = request.args.get('section', '')
    return render_template('msent.html', message=message, section=section)
@app.route('/video')
def serve_video():
    video_path = 'Aloevera.mp4'  # Replace with the actual path to your video file
    return send_file(video_path, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True)
