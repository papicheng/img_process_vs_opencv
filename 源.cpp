#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/opencv.hpp"
#include "opencv2/opencv_modules.hpp"
#include <iostream>
#include <io.h>
#include <thread>
#include <mutex>
#include <vector>
#include <queue>
#include <atomic>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

using namespace cv;
using namespace std;

void img_process(string path, string filename);
void getFiles(const string& path, vector<string>& files);

class ThreadPool {
public:
	ThreadPool(size_t);
	template<class F, class... Args>
	auto enqueue(F&& f, Args&&... args)
		->std::future<typename std::result_of<F(Args...)>::type>;
	~ThreadPool();
private:
	// need to keep track of threads so we can join them
	std::vector< std::thread > workers;
	// the task queue
	std::queue< std::function<void()> > tasks;

	// synchronization
	std::mutex queue_mutex;
	std::condition_variable condition;
	bool stop;
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads)
	: stop(false)
{
	for (size_t i = 0; i < threads; ++i)
		workers.emplace_back(
			[this]
			{
				for (;;)
				{
					std::function<void()> task;

					{
						std::unique_lock<std::mutex> lock(this->queue_mutex);
						this->condition.wait(lock,
							[this] { return this->stop || !this->tasks.empty(); });
						if (this->stop && this->tasks.empty())
							return;
						task = std::move(this->tasks.front());
						this->tasks.pop();
					}

					task();
				}
			}
			);
}

// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
-> std::future<typename std::result_of<F(Args...)>::type>
{
	using return_type = typename std::result_of<F(Args...)>::type;

	auto task = std::make_shared< std::packaged_task<return_type()> >(
		std::bind(std::forward<F>(f), std::forward<Args>(args)...)
		);

	std::future<return_type> res = task->get_future();
	{
		std::unique_lock<std::mutex> lock(queue_mutex);

		// don't allow enqueueing after stopping the pool
		if (stop)
			throw std::runtime_error("enqueue on stopped ThreadPool");

		tasks.emplace([task]() { (*task)(); });
	}
	condition.notify_one();
	return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
	{
		std::unique_lock<std::mutex> lock(queue_mutex);
		stop = true;
	}
	condition.notify_all();
	for (std::thread& worker : workers)
		worker.join();
}

int main()
{

	string path = "D:/img_repair/";
	//string filename = path + "/img/(0100-Y19-1)263881-264380-0032.jpg";
	vector<string> files;
	getFiles(path + "cbc", files);

	ThreadPool pool(50);
	int file_num = 0;
	while (file_num < files.size()) {
		if(files[file_num].size() < 15) continue;
		try {
			pool.enqueue(img_process, path, files[file_num]);
			file_num++;
		}
		catch (exception& e) {
			cout << "error: " << this_thread::get_id() << e.what() << endl;
		}
	}
	return 0;
}

void img_process(string path, string filename) {
	static int num = 0;
	Mat img = imread(filename, 1);
	int halfRow = img.rows / 2, halfCol = img.cols / 2;
	Mat img_lur;
	int m = 300;
	img_lur = img(Range(halfRow - m, halfRow + m), Range(halfCol - 3 * m, halfCol + 3 * m));
	resize(img_lur, img_lur, Size(), 0.2, 0.2);
	
	imwrite(path + "/cbc/img_lur0.jpg", img_lur);
	medianBlur(img_lur, img_lur, 31);
	resize(img_lur, img_lur, Size(img.cols, img.rows));
	resize(img_lur, img_lur, Size(), 0.01, 0.01);
	medianBlur(img_lur, img_lur, 31);
	resize(img_lur, img_lur, Size(img.cols, img.rows));
	resize(img_lur, img_lur, Size(), 0.01, 0.01);
	medianBlur(img_lur, img_lur, 31);
	resize(img_lur, img_lur, Size(img.cols, img.rows));
	imwrite(path + "/cbc/img_lur.jpg", img_lur);
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	Mat binary;
	int blockSize = 85;
	int constValue = 10;
	adaptiveThreshold(gray, binary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);
	//threshold(gray, binary, 150, 255, THRESH_BINARY_INV |THRESH_OTSU );
	imwrite(path + "cbc/binary0.jpg", binary);
	
	Mat erode_img;
	Mat structure_element = getStructuringElement(MORPH_RECT, Size(17, 17));
	erode(binary, erode_img, structure_element);
	//imwrite(path + "/cbc/erode0.jpg", erode_img);
	structure_element = getStructuringElement(MORPH_RECT, Size(23, 23));
	dilate(erode_img, erode_img, structure_element);
	Mat labels, centroids, stats;
	int n_labels = connectedComponentsWithStats(erode_img, labels, stats, centroids);
	uchar* p;
	int dis = 200;
	for (int i = 1; i < n_labels; ++i) {
		if (stats.at<int>(i, CC_STAT_AREA) < 5200 || (
			stats.at<int>(i, CC_STAT_LEFT) > dis && (stats.at<int>(i, CC_STAT_WIDTH) + stats.at<int>(i, CC_STAT_LEFT)) < (img.cols - dis) && stats.at<int>(i, CC_STAT_TOP) > dis && (stats.at<int>(i, CC_STAT_TOP) + stats.at<int>(i, CC_STAT_HEIGHT)) < (img.rows - dis)
			) ) {
			for (int row = stats.at<int>(i, CC_STAT_TOP); row < stats.at<int>(i, CC_STAT_TOP) + stats.at<int>(i, CC_STAT_HEIGHT); ++row) {
				p = erode_img.ptr<uchar>(row);
				for (int col = stats.at<int>(i, CC_STAT_LEFT); col < stats.at<int>(i, CC_STAT_LEFT) + stats.at<int>(i, CC_STAT_WIDTH); col++) {
					if (p[col] > 0) p[col] = 0;
				}
			}
		}
	}
	structure_element = getStructuringElement(MORPH_RECT, Size(7, 7));
	dilate(erode_img, erode_img, structure_element);
	imwrite(path + "/cbc/erode1.jpg", erode_img);
	bitwise_not(erode_img, erode_img);
	bitwise_and(erode_img, binary, erode_img);
	imwrite(path + "/cbc/erode.jpg", erode_img);

	for (int i = 0; i < erode_img.rows; ++i) {
		p = erode_img.ptr<uchar>(i);
		for (int j = 0; j < erode_img.cols; ++j) {
			if (p[j] > 0) {
				img_lur.at<Vec3b>(i, j)[0] = img.at<Vec3b>(i, j)[0];
				img_lur.at<Vec3b>(i, j)[1] = img.at<Vec3b>(i, j)[1];
				img_lur.at<Vec3b>(i, j)[2] = img.at<Vec3b>(i, j)[2];
			}
		}
	}
	int n = 15;
	img_lur = img_lur(Range(n, img_lur.rows - n), Range(n, img_lur.cols - n));
	erode_img = erode_img(Range(n, erode_img.rows - n), Range(n, erode_img.cols - n));
	img = img(Range(n, img.rows - n), Range(n, img.cols - n));
	//n_labels = connectedComponentsWithStats(erode_img, labels, stats, centroids);
	//for (int i = 1; i < n_labels; ++i) {
	//	//cout << stats.at<int>(i, CC_STAT_AREA) << endl;
	//	if (stats.at<int>(i, CC_STAT_AREA) > 200) {
	//		for (int row = stats.at<int>(i, CC_STAT_TOP); row < stats.at<int>(i, CC_STAT_TOP) + stats.at<int>(i, CC_STAT_HEIGHT); ++row) {
	//			p = erode_img.ptr<uchar>(row);
	//			for (int col = stats.at<int>(i, CC_STAT_LEFT); col < stats.at<int>(i, CC_STAT_LEFT) + stats.at<int>(i, CC_STAT_WIDTH); col++) {
	//				img_lur.at<Vec3b>(row, col)[0] = img.at<Vec3b>(row, col)[0];
	//				img_lur.at<Vec3b>(row, col)[1] = img.at<Vec3b>(row, col)[1];
	//				img_lur.at<Vec3b>(row, col)[2] = img.at<Vec3b>(row, col)[2];
	//			}
	//		}
	//	}
	//}
	string img_path = path + "cbc/img" + to_string(num++) + ".jpg";
	imwrite(img_path, img_lur);
}

void getFiles(const string& path, vector<string>& files)
{
	long long hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	int i = 0;
	if ((hFile = _findfirst(p.assign(path).append("/*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("/").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("/").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}







