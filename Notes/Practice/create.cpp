#include<bits/stdc++.h>
using namespace std;

int main() {
	string a;
	long long int i = 0;
	while(cin >> a, a[0] != '0') {
		cout << a << " ";
		i++;
		if(i == 2) cout << endl;
		i%3;
	}
}