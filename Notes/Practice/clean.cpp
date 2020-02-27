#include<bits/stdc++.h>
using namespace std;

int main() {
	string a;
	long long int i = 0;
	while(cin >> a, a[0] != '0') {
		if(a[0] != '[') {
			i++;
			cout << a << " ";
		}
		if(i == 3) {
			cout << endl;
			i = 0;
		}
	}
}