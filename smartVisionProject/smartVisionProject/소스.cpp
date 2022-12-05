#include<iostream> //������� ����
#include<opencv2/opencv.hpp> //������� ����
#include<string> //������� ����
#include<time.h> //������� ����

using namespace cv::dnn; //�̸����� ���
using namespace std; //�̸����� ���
using namespace cv; //�̸����� ���

double maxVal; //���� ����
String className; //���� ����
Mat savedImage, roiImage, prob; //��ü ����
Net net; //��ü ����
Point ptOld, maxLoc; //��ü ����

vector<string> classNames; //�迭 ����
vector<string> nutrient = { "calorie","carbohydrate","sugars","protein","fat","saturated fat","trans fat",
"cholesterol","salt","* Percent nutrient reference value : Percentage of daily nutrient reference value" }; //�迭 ����
string calorie, carbohydrate, sugars, protein, fat, saturatedFat, transFat, cholesterol, salt; //���� ����
string* s[] = { &calorie, &carbohydrate, &sugars, &protein, &fat, &saturatedFat,&transFat,&cholesterol,
&salt }; //�迭 ����

void train(); //�н� �Լ�
void imageSave(Mat f); //�̹��� ���� �Լ�
void on_mouse(int event, int x, int y, int flags, void*); //���콺 �̺�Ʈ
void nutrientInfo(); //����� ���� ��� �Լ�

int main() { //�����Լ�
	VideoCapture cap(0); //ī�޶� ����

	if (!cap.isOpened()) { //������ ������
		cerr << "Camrea open failed!" << endl; //���� �޽��� ���
		return -1; //�� ����
	}

	Mat frame; //mat ��ü ����
	double fps = cap.get(CAP_PROP_FPS); //�ʴ� ������ ��
	int num_frames = 1; //ĸ���� ������ ��
	clock_t start; //���� �ð� ����
	clock_t end; //������ �ð� ����

	cout << "s�� ������ ����, esc�� ������ ����" << endl; //�޽��� ���

	while (true) { //���� �ݺ���
		int k = waitKey(10); //Ű�Է� ���

		cap >> frame; //������ �޾ƿ���
		if (frame.empty()) break; //�������� ����ִٸ� ����

		imshow("frame", frame); //�ǽð� ���� ���

		if (k == 27) break; //escŰ ������ ����
		else if (k == 's') { //sŰ ������
			cout << "s�� �����̽��ϴ�. �̹����� ����˴ϴ�." << endl; //�޽��� ���

			imageSave(frame); //�̹��� ���� �Լ� ����
			start = clock();//���� �ð�
			train(); //�н�&���� �Լ� ����
			nutrientInfo(); //����� ���� ��� �Լ� ����
			end = clock(); //������ �ð�

			double seconds = (double(end) - double(start)) / double(CLOCKS_PER_SEC); //����ð� ���ϱ�(��)
			cout << "�ɸ� �ð� : " << seconds << " seconds" << endl; //����ð� ����� ���
			double fpsLive = double(num_frames) / double(seconds); //�ǽð� fps ���ϱ�
			cout << "fps : " << fpsLive << endl; //�ǽð� fps ����� ���
		}
	}
	destroyAllWindows(); //��� ������ â ����

	return 0; //����
}

void imageSave(Mat f) { //�̹��� ���� �Լ�
	imwrite("savedImage.jpg", f); //�������� �̹����� ����

	savedImage = imread("savedImage.jpg"); //�̹����� �ҷ���
	imshow("savedImage", savedImage); //����� �̹��� ���

	cout << "��ǰ�� �巡���ϼ���" << endl; //�޽��� ���
	setMouseCallback("savedImage", on_mouse); //���콺 �̺�Ʈ ����
	if (waitKey() == ' ') return; //�����̽��ٸ� ������ �Լ�����
}

void on_mouse(int event, int x, int y, int flags, void*) { //���콺 �̺�Ʈ �Լ�
	switch (event) { //swtich��
	case EVENT_LBUTTONDOWN: //���콺 ���� ��ư Ŭ����
		ptOld = Point(x, y); //����Ʈ ����
		break; //����
	case EVENT_LBUTTONUP: //���콺 ���� ��ư ������ ����
		roiImage = savedImage(Rect(ptOld, Point(x, y))).clone(); //����� �̹������� �巡���� �κ��� ����
		imshow("roiImage", roiImage); //�κ� �̹��� ���
		imwrite("roiImage.jpg", roiImage); //�κ� �̹����� ����
		cout << "space�� ������ ����� ���� ���" << endl; //�޽��� ���
		break; //����
	default: //�� �ܿ�
		break; //����
	}
}

void train() { //�н� �Լ�
	classNames = { "ghana","pefero","abccookie","cereal","magaret","cancho","chaltteokpie",
		"chambungeobbang","freshberry","custard","chocosongi","ohyesmini","eggsnack","ivy","chocoheim",
		"bbotto","cookdas","crownsando" }; //Ŭ���� �̸� ����
	
	net = readNet("frozen_model.pb"); //������ �ҷ���
	if (net.empty()) { cerr << "Network load failed!" << endl; return; } //������ ������ ���� ��� �� ����

	Mat inputBlob = blobFromImage(roiImage, 1 / 127.5, Size(224, 224), -1.0); //�̹��� ��ó��
	net.setInput(inputBlob); //����
	prob = net.forward(); //forward ���� ����

	minMaxLoc(prob, NULL, &maxVal, NULL, &maxLoc); //minmaxloc�Լ� ����
	String str = classNames[maxLoc.x] + format("(%4.2lf%%)", maxVal * 100); //Ŭ���� �̸��� �νķ� ����
	putText(roiImage, str, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 128, 128)); //���

	imshow("imgResult", roiImage); //�κп��� ����
}

void nutrientInfo() { //����� ���� ��� �Լ�
	className = classNames[maxLoc.x]; //Ŭ���� �̸��� �߷а� ����
	cout << className << "�� ����� ������ ����մϴ�." << endl; //�޽��� ���

	switch (maxLoc.x) { //Ŭ���� �̸��� ���� �� ������ ����� ���� ����
	case 0: 
		calorie = "195kcal", carbohydrate = "19g(6%)", sugars = "18g(18%)", protein = "3g(5%)", fat = "12g(24%)",
			saturatedFat = "7g(47%)", transFat = "0g", cholesterol = "5mg(2%)", salt = "25mg(1%)";
		break;
	case 1:
		calorie = "250kcal", carbohydrate = "36g(11%)", sugars = "13g(13%)", protein = "6g(11%)", fat = "9g(17%)",
			saturatedFat = "6g(40%)", transFat = "0g", cholesterol = "less than 5mg(1%)", salt = "210mg(11%)";
		break;
	case 2:
		calorie = "175kcal", carbohydrate = "19g(6%)", sugars = "11g(11%)", protein = "2g(4%)", fat = "10g(19%)",
			saturatedFat = "5g(33%)", transFat = "0g", cholesterol = "5mg(2%)", salt = "90mg(5%)";
		break;
	case 3:
		calorie = "215kcal", carbohydrate = "26g(8%)", sugars = "15g(15%)", protein = "3g(5%)", fat = "11g(20%)",
			saturatedFat = "8g(53%)", transFat = "0g", cholesterol = "less than 5mg(1%)", salt = "75mg(4%)";
		break;
	case 4:
		calorie = "220kcal", carbohydrate = "26g(8%)", sugars = "11g(11%)", protein = "2g(4%)", fat = "12g(24%)",
			saturatedFat = "6g(40%)", transFat = "less than 0.5g", cholesterol = "10mg(3%)", salt = "120mg(6%)";
		break;
	case 5:
		calorie = "190kcal", carbohydrate = "27g(8%)", sugars = "12g(12%)", protein = "3g(5%)", fat = "8g(15%)",
			saturatedFat = "4.4g(29%)", transFat = "0g", cholesterol = "0mg(0%)", salt = "110mg(6%)";
		break;
	case 6:
		calorie = "140kcal", carbohydrate = "26g(8%)", sugars = "11g(11%)", protein = "0.9g(2%)", fat = "3.8g(7%)",
			saturatedFat = "2.7g(18%)", transFat = "0g", cholesterol = "2mg(1%)", salt = "105mg(5%)";
		break;
	case 7:
		calorie = "121kcal", carbohydrate = "17g(5%)", sugars = "10g(10%)", protein = "2g(4%)", fat = "5g(9%)",
			saturatedFat = "2.3g(15%)", transFat = "0g", cholesterol = "20mg(7%)", salt = "70mg(4%)";
		break;
	case 8:
		calorie = "127kcal", carbohydrate = "15g(5%)", sugars = "9g(9%)", protein = "1g(2%)", fat = "7g(13%)",
			saturatedFat = "3.4g(23%)", transFat = "0g", cholesterol = "16mg(5%)", salt = "50mg(3%)";
		break;
	case 9:
		calorie = "208kcal", carbohydrate = "22g(7%)", sugars = "12g(12%)", protein = "3g(5%)", fat = "12g(24%)",
			saturatedFat = "4.9g(33%)", transFat = "0g", cholesterol = "55mg(18%)", salt = "60mg(3%)";
		break;
	case 10:
		calorie = "195kcal", carbohydrate = "22g(7%)", sugars = "14g(14%)", protein = "2g(4%)", fat = "11g(22%)",
			saturatedFat = "6g(40%)", transFat = "0g", cholesterol = "0mg(0%)", salt = "80mg(4%)";
		break;
	case 11:
		calorie = "145kcal", carbohydrate = "16g(5%)", sugars = "10g(10%)", protein = "2g(4%)", fat = "8g(15%)",
			saturatedFat = "4.6g(31%)", transFat = "0.06g", cholesterol = "15mg(5%)", salt = "95mg(5%)";
		break;
	case 12:
		calorie = "315kcal", carbohydrate = "52g(16%)", sugars = "24g(24%)", protein = "4g(7%)", fat = "10g(20%)",
			saturatedFat = "6g(40%)", transFat = "0g", cholesterol = "25mg(8%)", salt = "140mg(7%)";
		break;
	case 13:
		calorie = "170kcal", carbohydrate = "28g(8%)", sugars = "1g(1%)", protein = "4g(7%)", fat = "4.9g(10%)",
			saturatedFat = "3g(20%)", transFat = "0g", cholesterol = "0mg(0%)", salt = "280mg(14%)";
		break;
	case 14:
		calorie = "170kcal", carbohydrate = "18g(5%)", sugars = "10g(10%)", protein = "3g(5%)", fat = "10g(20%)",
			saturatedFat = "5g(33%)", transFat = "0g", cholesterol = "less than 5mg(1%)", salt = "50mg(3%)";
		break;
	case 15:
		calorie = "120kcal", carbohydrate = "14g(4%)", sugars = "3g(3%)", protein = "2g(4%)", fat = "6g(12%)",
			saturatedFat = "3.8g(25%)", transFat = "0g", cholesterol = "15mg(5%)", salt = "120mg(6%)";
		break;
	case 16:
		calorie = "175kcal", carbohydrate = "19g(6%)", sugars = "11g(11%)", protein = "2g(4%)", fat = "110g(20%)0",
			saturatedFat = "7g(17%)", transFat = "0g", cholesterol = "0mg(0%)", salt = "55mg(3%)";
		break;
	case 17:
		calorie = "100kcal", carbohydrate = "13g(4%)", sugars = "5g(5%)", protein = "1g(2%)", fat = "5g(10%)",
			saturatedFat = "3.2g(21%)", transFat = "0g", cholesterol = "less than 5mg(1%)", salt = "50mg(3%)";
		break;
	default:
		break;
	}

	Mat info(500, 650, CV_8UC3, Scalar(255, 255, 255)); //�� ��� ����

	rectangle(info, Rect(Point(0, 0), Point(info.cols, info.rows)), Scalar(0, 0, 0), 2); //�׵θ� �簢��
	line(info, Point(info.cols / 2, info.rows * 1 / 11), Point(info.cols / 2, info.rows * 10 / 11), 
		Scalar(0, 0, 0), 1); //ǥ ���μ� �����
	for (int i = 1; i <= 11; i++) { //for��
		line(info, Point(0, info.rows * i / 11), Point(info.cols, info.rows * i / 11), Scalar(0, 0, 0), 1); //ǥ ���μ� �����
	}

	int fontFace = FONT_HERSHEY_PLAIN; //��Ʈ ����
	double fontScale = 3.0; //�۾� ũ��
	int thickness = 3; //�۾� ����

	Size sizeText[10], nameSizeText, tailSizeText; //��ü ����
	Size sizeImg = info.size(); //info �̹��� ũ�� ���� 

	nameSizeText = getTextSize(className, fontFace, fontScale, thickness, 0); //��ǰ �̸� ũ�� ����
	Point nameOrg((sizeImg.width - nameSizeText.width) / 2, (sizeImg.height + nameSizeText.height) / 12); //����Ʈ ����
	putText(info, className, nameOrg, fontFace, fontScale, Scalar(0, 0, 0), thickness); //��ǰ �̸� ���

	thickness = 1; //�۾� ���� ����
	tailSizeText = getTextSize(nutrient[9], fontFace, 0.9, thickness, 0); //�޽��� ũ�� ����
	Point tailOrg((sizeImg.width - tailSizeText.width) / 2, (sizeImg.height + tailSizeText.height) * 11 / 11.8); //����Ʈ ����
	putText(info, nutrient[9], tailOrg, fontFace, 0.9, Scalar(0, 0, 0), thickness); //�޽��� ���

	for (int i = 0; i < 9; i++) { //�ݺ���
		fontFace = FONT_HERSHEY_PLAIN; //��Ʈ ����
		fontScale = 2.0; //�۾� ũ��
		String n = *s[i]; //���� �� ����

		sizeText[i] = getTextSize(nutrient[i], fontFace, fontScale, thickness, 0); //����� ���и� �۾� ũ�� ����
		Point nutrientOrg((sizeImg.width - sizeText[i].width) / 50, (sizeImg.height + sizeText[i].height)* (i + 1.9) / 11.7); //����Ʈ ����
		putText(info, nutrient[i], nutrientOrg, fontFace, fontScale, Scalar(0, 0, 0), thickness); //���и� ���
		
		sizeText[i] = getTextSize(n, fontFace, fontScale, thickness, 0);//����� ���� �۾� ũ�� ����
		Point percentOrg((sizeImg.width - sizeText[i].width) * 49 / 50, (sizeImg.height + sizeText[i].height)* (i + 1.9) / 11.7); //����Ʈ ����
		putText(info, n, percentOrg, fontFace, fontScale, Scalar(0, 0, 0), thickness); //����� ���� ���
	}

	imshow("info", info); //���� ���â
	
	cout << "���� ��� �Ϸ�. esc�� ������ ����, �ٽ� �Ϸ��� s�� ��������.\n" << endl; //�޽��� ���
}