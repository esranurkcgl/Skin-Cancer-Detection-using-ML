# The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.
from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import SkinCancerDetection as SCD

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def runhome():

    return render_template("home.html")

@app.route("/showresult", methods=["GET", "POST"])
def show():
    pic = request.files["pic"]
    inputimg = Image.open(pic)
    inputimg = inputimg.resize((100,100))
    img = np.array(inputimg).reshape(-1, 100, 100, 3)
    result = SCD.model.predict(img)

    result = result.tolist()
    print(result)
    max_prob = max(result[0])
    class_ind = result[0].index(max_prob)
    print(class_ind)
    result = SCD.classes[class_ind]

    if class_ind == 0:
        info = "Melanocytic nevi; insanların büyük bölümünde bir ya da birkaç tane olabilen, melanin üreten hücrelerin (melanosit) iyi huylu tümörüdür."
    elif class_ind == 1:
        info = "Melanoma; Deriye rengini veren melanin adlı renk pigmentleri, melanosit denilen cilt hücreleri tarafından üretilir. Bu hücrelerin kontrolsüz bir şekilde bölünüp çoğalması melanom veya melanoma olarak tanımlanır"
    elif class_ind == 2:
        info = "Benign keratosis-like lesions; liken planus benzeri keratoz olarak da bilinen, derinin malign lezyonlarıyla karıştırılabilen sık bir antitedir. Genellikle 35-65 yaş arası kadınlarda, yüz ve üst gövdede, asemptomatik soliter pembe-kırmızı-kahverengi papül veya hafif endüre plak olarak izlenir."
    elif class_ind == 3:
        info = "Basal cell carcinoma; Bazal hücreli karsinom, bir cilt kanseri türüdür. BHK olarak kısaltılır. Deride bulunan eski hücrelerin ölmesiyle onların yerine yenilerini üreten, bazal hücre olarak adlandırılan hücre tipinde başlar. BHKfarklı şekillerde olabilmekle birlikte sıklıkla ciltte hafif şeffaf bir yumru olarak belirir. "
    elif class_ind == 4:
        info = "Actinic keratoses; Aktinik keratoz, uzun süreli kontrolsuz güneşe maruz kalmaya bağlı olarak en çok güneş gören bölgelerde görülen deride anormal hücre gelişimini yansıtan deri değişiklikleridir. En sık yüzeyi pürtüklü yama şeklinde görülürler. Aktinik keratozların düşük oranda deri kanserine dönüşme riski vardır. "
    elif class_ind == 5:
        info = "Vascular lesion; Vaskuler lezyon nedir? Ciltteki kılcal damarlar yapılarının belirginliğinin artması; kılcal damarların çatlaması veya genişlemesi sonucu ciltte mor, kırmızı renkte hat/hatlar oluşturması durumu damarsal lezyon veya vasküler lezyon olarak tanımlanır. "
    elif class_ind == 6:
        info = " Dermatofibroma; Dermatofibroma orta yaş erişkinlerde oldukça sık görülen, benign, fibrohistiyositik bir deri neoplazisidir. Genellikle alt ekstremitede yerleşen sert, tek veya multipl, papül, plak ya da nodül şeklindeki lezyonlarla karakterizedir."

    return render_template("result.html", result=result, info=info)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
