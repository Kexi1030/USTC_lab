#include<vector>
#include<algorithm>
#include<stdio.h>
#include<stdlib.h>
#include<fstream>
#include<pthread.h>
#include<iostream>

using namespace std;

#define MAXM 5
#define MAXC 5

typedef struct Node {
	int M, C, B;
	int g, h, f;
	int  parent;//记录父节点在tree中的下标
}Node;

vector<Node> open;
vector<Node> close;
vector<Node> tree;

Node boat[8] = {
	{ 0,1,1,0,0,0,-1 },
	{ 0,2,1,0,0,0,-1 },
	{ 0,3,1,0,0,0,-1 },
	{ 1,0,1,0,0,0,-1 },
	{ 1,1,1,0,0,0,-1 },
	{ 2,0,1,0,0,0,-1 },
	{ 2,1,1,0,0,0,-1 },
	{ 3,0,1,0,0,0,-1 }
};


void show_open() {//显示open表
	vector<Node>::iterator ite;
	ite = open.end() - 1;
	for (; ite >= open.begin(); ite--) {
		printf("(%d,%d,%d,%d,%d,%d)\n", (*ite).M, (*ite).C, (*ite).B, (*ite).g, (*ite).h, (*ite).f);
		//writeintotxt();
		if (ite == open.begin()) break;
	}
}


void show_close() {//显示close表
	vector<Node>::iterator ite;
	ite = close.begin();
	for (; ite < close.end(); ite++) {
		printf("(%d,%d,%d,%d,%d,%d)\n", (*ite).M, (*ite).C, (*ite).B, (*ite).g, (*ite).h, (*ite).f);
	}
}


bool operator ==(Node a, Node b) { // 操作符重写
	return(a.M == b.M && a.C == b.C && a.B == b.B);
}


bool exit_open(Node p) {//判断节点是否存在open表中
	if (find(open.begin(), open.end(), p) == open.end()) return false;
	else return true;
}


bool exit_close(Node p) {//判断节点是否存在close表中
	if (find(close.begin(), close.end(), p) == close.end()) return false;
	else return true;
}


bool comp(Node a, Node b) {
	return a.f > b.f;//根据估价函数值降序排列
}


void add_open(Node p) {//open表添加
	open.push_back(p);
	sort(open.begin(), open.end(), comp);//默认升序排列，这里comp按降序排列
}


void add_close(Node p) {//close表添加
	close.push_back(p);
}


void out_open() {//open表删除
	open.pop_back();
}


bool judge_Node(Node p) {//判断状态p是否合法
	if (p.M > MAXM || p.C > MAXC || p.M < 0 || p.C < 0)//不在范围内。不合法
		return false;
	/*if (((p.M >= p.C) && (MAXM - p.M >= MAXC - p.C)) || (p.M == MAXM) || (p.M == 0))
		return true;
	return false;*/
	else if (p.M != 0 && p.M < p.C)//左岸传教士人数不为0 并且小于野人数
		return false;
	else if (MAXM - p.M != 0 && MAXM - p.M < MAXC - p.C)//右岸传教士人数不为0 error
		return false;
	else return true;
}


void expand(Node p) {//对节点p进行扩展
	Node q;
	printf("************************************************************************************\n");
	printf("\t\t\t对结点: ");
	printf("( %d %d %d )进行扩展\n", p.M, p.C, p.B);
	for (int i = 0; i < 8; i++) {
		if (p.B == 1) {//p船在左岸 
			q.M = p.M - boat[i].M;
			q.C = p.C - boat[i].C;
			q.B = p.B - boat[i].B;
		}
		else {//p船在右岸
			q.M = p.M + boat[i].M;
			q.C = p.C + boat[i].C;
			q.B = p.B + boat[i].B;
		}
		if (judge_Node(q) && !exit_open(q) && !exit_close(q)) {//避免死循环
			q.g = p.g + 1;
			q.h = q.M + q.C - 2 * q.B;
			q.f = q.g + q.h;
			int pos = find(tree.begin(), tree.end(), p) - tree.begin();
			q.parent = pos;
			printf("扩展出新的子结点:");
			printf("(%d,%d,%d,%d,%d,%d)\n", q.M, q.C, q.B, q.g, q.h, q.f);
			add_open(q);
			tree.push_back(q);
		}
		else {
			printf("\t******节点(%d,%d,%d)不满足条件，扩展失败******\n", q.M, q.C, q.B);
		}
	}
	//printf("\t\t=======================================================\n");
	add_close(p);
	printf("\t\t\t\t******open表状态******\n");
	show_open();
	printf("\t\t\t\t******close表状态******\n");

	show_close();
	printf("************************************************************************************\n");
}


bool destination(Node p) {//判断p是否为目标节点
	if (p.M == 0 && p.C == 0 && p.B == 0) return true;
	else return false;
}


Node solve() {
	Node p{ 5, 5, 1, 0, 10, 10,-1 };
	open.push_back(p);
	tree.push_back(p);
	char c;
	Node x;
	while (open.size() != 0) {
		x = *(open.end() - 1);//从open表中取出一个进行扩展
		if (destination(x)) return x;//如果是目标状态  则结束
		out_open();//从open中删除
		expand(x);//扩展该结点
		//getchar();
	}
	//return NULL;
}


void path(Node p) {
	vector<Node> temp;
	while (p.parent != -1) {
		temp.push_back(p);
		p = tree[p.parent];
	}
	temp.push_back(p);
	vector<Node>::iterator ite1 = temp.end() - 1;
	for (; ite1 >= temp.begin(); ite1--) {
		printf("(%d,%d,%d,%d,%d,%d)\n", (*ite1).M, (*ite1).C, (*ite1).B, (*ite1).g, (*ite1).h, (*ite1).f);
		if (ite1 == temp.begin()) break;
	}
}


void* StartSearch(void* args)
{
	Node goal = solve();
	printf("求得M-C问题的解如下所示：\n");
	path(goal);
	cout << "\n\n线程退出......\n" << endl;
	return 0;
}


int main() {
	pthread_t Allthreads[1];
	printf("线程开始......\n");
	int ref = pthread_create(&Allthreads[0], NULL, StartSearch, NULL);
	if (ref != 0)
	{
		cout << "pthread_create error!!!!!!" << endl;
	}
	// 等线程退出后，进程才会结束
	pthread_exit(NULL);
}