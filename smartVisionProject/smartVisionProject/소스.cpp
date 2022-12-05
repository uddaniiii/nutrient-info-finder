#include<iostream> //헤더파일 포함
#include<opencv2/opencv.hpp> //헤더파일 포함
#include<string> //헤더파일 포함
#include<time.h> //헤더파일 포함

using namespace cv::dnn; //이름공간 사용
using namespace std; //이름공간 사용
using namespace cv; //이름공간 사용

double maxVal; //변수 선언
String className; //변수 선언
Mat savedImage, roiImage, prob; //객체 선언
Net net; //객체 선언
Point ptOld, maxLoc; //객체 선언

vector<string> classNames; //배열 선언
vector<string> nutrient = { "calorie","carbohydrate","sugars","protein","fat","saturated fat","trans fat",
"cholesterol","salt","* Percent nutrient reference value : Percentage of daily nutrient reference value" }; //배열 선언
string calorie, carbohydrate, sugars, protein, fat, saturatedFat, transFat, cholesterol, salt; //변수 선언
string* s[] = { &calorie, &carbohydrate, &sugars, &protein, &fat, &saturatedFat,&transFat,&cholesterol,
&salt }; //배열 선언

void train(); //학습 함수
void imageSave(Mat f); //이미지 저장 함수
void on_mouse(int event, int x, int y, int flags, void*); //마우스 이벤트
void nutrientInfo(); //영양소 정보 출력 함수

int main() { //메인함수
	VideoCapture cap(0); //카메라 접근

	if (!cap.isOpened()) { //열리지 않으면
		cerr << "Camrea open failed!" << endl; //에러 메시지 출력
		return -1; //후 종료
	}

	Mat frame; //mat 객체 생성
	double fps = cap.get(CAP_PROP_FPS); //초당 프레임 수
	int num_frames = 1; //캡쳐할 프레임 수
	clock_t start; //시작 시간 변수
	clock_t end; //끝나는 시간 변수

	cout << "s를 누르면 저장, esc를 누르면 종료" << endl; //메시지 출력

	while (true) { //무한 반복문
		int k = waitKey(10); //키입력 대기

		cap >> frame; //프레임 받아오기
		if (frame.empty()) break; //프레임이 비어있다면 멈춤

		imshow("frame", frame); //실시간 영상 출력

		if (k == 27) break; //esc키 누르면 종료
		else if (k == 's') { //s키 누르면
			cout << "s를 누르셨습니다. 이미지가 저장됩니다." << endl; //메시지 출력

			imageSave(frame); //이미지 저장 함수 실행
			start = clock();//시작 시간
			train(); //학습&예측 함수 실행
			nutrientInfo(); //영양소 정보 출력 함수 실행
			end = clock(); //끝나는 시간

			double seconds = (double(end) - double(start)) / double(CLOCKS_PER_SEC); //경과시간 구하기(초)
			cout << "걸린 시간 : " << seconds << " seconds" << endl; //경과시간 결과값 출력
			double fpsLive = double(num_frames) / double(seconds); //실시간 fps 구하기
			cout << "fps : " << fpsLive << endl; //실시간 fps 결과값 출력
		}
	}
	destroyAllWindows(); //모든 윈도우 창 지움

	return 0; //종료
}

void imageSave(Mat f) { //이미지 저장 함수
	imwrite("savedImage.jpg", f); //프레임을 이미지로 저장

	savedImage = imread("savedImage.jpg"); //이미지를 불러옴
	imshow("savedImage", savedImage); //저장된 이미지 출력

	cout << "상품을 드래그하세요" << endl; //메시지 출력
	setMouseCallback("savedImage", on_mouse); //마우스 이벤트 실행
	if (waitKey() == ' ') return; //스페이스바를 누르면 함수종료
}

void on_mouse(int event, int x, int y, int flags, void*) { //마우스 이벤트 함수
	switch (event) { //swtich문
	case EVENT_LBUTTONDOWN: //마우스 왼쪽 버튼 클릭시
		ptOld = Point(x, y); //포인트 저장
		break; //종료
	case EVENT_LBUTTONUP: //마우스 왼쪽 버튼 눌렀다 뗄시
		roiImage = savedImage(Rect(ptOld, Point(x, y))).clone(); //저장된 이미지에서 드래그한 부분을 추출
		imshow("roiImage", roiImage); //부분 이미지 출력
		imwrite("roiImage.jpg", roiImage); //부분 이미지로 저장
		cout << "space를 누르면 영양소 정보 출력" << endl; //메시지 출력
		break; //종료
	default: //그 외에
		break; //종료
	}
}

void train() { //학습 함수
	classNames = { "ghana","pefero","abccookie","cereal","magaret","cancho","chaltteokpie",
		"chambungeobbang","freshberry","custard","chocosongi","ohyesmini","eggsnack","ivy","chocoheim",
		"bbotto","cookdas","crownsando" }; //클래스 이름 저장
	
	net = readNet("frozen_model.pb"); //모델파일 불러옴
	if (net.empty()) { cerr << "Network load failed!" << endl; return; } //모델파일 없으면 에러 출력 후 종료

	Mat inputBlob = blobFromImage(roiImage, 1 / 127.5, Size(224, 224), -1.0); //이미지 전처리
	net.setInput(inputBlob); //저장
	prob = net.forward(); //forward 연산 진행

	minMaxLoc(prob, NULL, &maxVal, NULL, &maxLoc); //minmaxloc함수 실행
	String str = classNames[maxLoc.x] + format("(%4.2lf%%)", maxVal * 100); //클래스 이름과 인식률 저장
	putText(roiImage, str, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 128, 128)); //출력

	imshow("imgResult", roiImage); //부분영역 추출
}

void nutrientInfo() { //영양소 정보 출력 함수
	className = classNames[maxLoc.x]; //클래스 이름에 추론값 저장
	cout << className << "의 영양소 정보를 출력합니다." << endl; //메시지 출력

	switch (maxLoc.x) { //클래스 이름에 따라 각 과자의 영양소 정보 저장
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

	Mat info(500, 650, CV_8UC3, Scalar(255, 255, 255)); //흰 배경 생성

	rectangle(info, Rect(Point(0, 0), Point(info.cols, info.rows)), Scalar(0, 0, 0), 2); //테두리 사각형
	line(info, Point(info.cols / 2, info.rows * 1 / 11), Point(info.cols / 2, info.rows * 10 / 11), 
		Scalar(0, 0, 0), 1); //표 세로선 만들기
	for (int i = 1; i <= 11; i++) { //for문
		line(info, Point(0, info.rows * i / 11), Point(info.cols, info.rows * i / 11), Scalar(0, 0, 0), 1); //표 가로선 만들기
	}

	int fontFace = FONT_HERSHEY_PLAIN; //폰트 선택
	double fontScale = 3.0; //글씨 크기
	int thickness = 3; //글씨 굵기

	Size sizeText[10], nameSizeText, tailSizeText; //객체 선언
	Size sizeImg = info.size(); //info 이미지 크기 저장 

	nameSizeText = getTextSize(className, fontFace, fontScale, thickness, 0); //상품 이름 크기 저장
	Point nameOrg((sizeImg.width - nameSizeText.width) / 2, (sizeImg.height + nameSizeText.height) / 12); //포인트 선언
	putText(info, className, nameOrg, fontFace, fontScale, Scalar(0, 0, 0), thickness); //상품 이름 출력

	thickness = 1; //글씨 굵기 선언
	tailSizeText = getTextSize(nutrient[9], fontFace, 0.9, thickness, 0); //메시지 크기 저장
	Point tailOrg((sizeImg.width - tailSizeText.width) / 2, (sizeImg.height + tailSizeText.height) * 11 / 11.8); //포인트 선언
	putText(info, nutrient[9], tailOrg, fontFace, 0.9, Scalar(0, 0, 0), thickness); //메시지 출력

	for (int i = 0; i < 9; i++) { //반복문
		fontFace = FONT_HERSHEY_PLAIN; //폰트 선택
		fontScale = 2.0; //글씨 크기
		String n = *s[i]; //변수 값 저장

		sizeText[i] = getTextSize(nutrient[i], fontFace, fontScale, thickness, 0); //영양소 성분명 글씨 크기 저장
		Point nutrientOrg((sizeImg.width - sizeText[i].width) / 50, (sizeImg.height + sizeText[i].height)* (i + 1.9) / 11.7); //포인트 선언
		putText(info, nutrient[i], nutrientOrg, fontFace, fontScale, Scalar(0, 0, 0), thickness); //성분명 출력
		
		sizeText[i] = getTextSize(n, fontFace, fontScale, thickness, 0);//영양소 비율 글씨 크기 저장
		Point percentOrg((sizeImg.width - sizeText[i].width) * 49 / 50, (sizeImg.height + sizeText[i].height)* (i + 1.9) / 11.7); //포인트 선언
		putText(info, n, percentOrg, fontFace, fontScale, Scalar(0, 0, 0), thickness); //영양소 비율 출력
	}

	imshow("info", info); //정보 출력창
	
	cout << "정보 출력 완료. esc를 누르면 종료, 다시 하려면 s를 누르세요.\n" << endl; //메시지 출력
}