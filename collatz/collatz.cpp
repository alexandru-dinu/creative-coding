#include <iostream>
#include <unordered_map>
#include <stdlib.h>

using namespace std;
using ll = long long;

// TODO: we may want a mapping from n to its sequence towards 1
unordered_map<ll, ll> cache;

ll collatz(ll n)
{
    if (cache.find(n) != cache.end())
        return cache[n];

    if (n <= 1)
        return 0;

    ll next;

    if (n & 1)
        next = 3 * n + 1;
    else
        next = n / 2;

    cache[n] = 1 + collatz(next);

    return cache[n];
}

int main(int argc, const char* argv[])
{
    if (argc != 2) {
        cout << "usage: ./collatz <LIMIT>\n";
        return -1;
    }

    ll limit = atol(argv[1]);

    for (int i = 1; i <= limit; ++i)
        collatz(i);

    return 0;
}
