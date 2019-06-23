// Copyright (C) 2019 Zhixian MA <zx@mazhixian.me>
// cpp notebook for vector
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

using namespace std;

// A template function
template <typename T>
void printVector(const vector<T>& inVec)
{
    for (auto element=inVec.cbegin(); 
         element != inVec.cend();
         ++element)
    {
        cout << *element << ' ';
    }
    cout << endl;
}

int main()
{
    // Initialization
    vector<int> intArray (10);

    // output the array
    vector<int>::const_iterator iterArray;

    cout << intArray[0] << endl;
    cout << intArray[1] << endl;

    // insert a new element
    intArray.push_back(10);

    cout << intArray.size() <<endl;

    // Init string array?
    vector<string> stringArray = {"a","he","she"};
    //cout << stringArray[2] << endl;
    printVector(stringArray);

    // sort?
    vector<int> sortArray{1,3,2,5,4};
    sortArray.insert(sortArray.begin()+2, 10);
    for (auto element = sortArray.cbegin();
         element != sortArray.cend();
         ++element)
    {
        cout << *element << ' '; 
    }
    cout << endl;
    // another method
    for (size_t index=0; index < sortArray.size(); index++)
    {
        cout << "Element[" << index << "] is ";
        cout << sortArray.at(index) << endl;
    }
    /* visit with pointer and iterator
    auto elementP = sortArray.cbegin();
    while (elementP != sortArray.end())
    {
        size_t index = distance(sortArray.cbegin(), elementP);
        cout << "Element[" << index << "] is ";
        cout << *elementP << endl;

        elementP ++;
    }*/
    cout << "Capacity " << sortArray.capacity() << endl;

    // clear
    sortArray.clear();
    cout << "Size: " << sortArray.size() << endl;
    cout << "Capacity: " << sortArray.capacity() << endl;

    return 0;
}