def count_element(arr, x, st, en):
    """
    It gets "SORTED" arr and return count of element x in array in log(len(arr))
    """
    i = first_element(arr, st, en-1, x, st, en)
    if i == -1:
        return i
    j = last_element(arr, i, en-1, x, st, en);    
    return j-i+1
        
def first_element(arr, low, high, x, st, en):
    if high >= low:
        mid = (low + high)//2     
        if (mid == st or x > arr[mid-1]) and arr[mid] == x:
            return mid
        elif x > arr[mid]:
            return first_element(arr, (mid + 1), high, x, st, en)
        else:
            return first_element(arr, low, (mid -1), x, st, en)
    return -1

def last_element(arr, low, high, x, st, en):
    if high >= low:
        mid = (low + high)//2
        if(mid == en-1 or x < arr[mid+1]) and arr[mid] == x :
            return mid
        elif x < arr[mid]:
            return last_element(arr, low, (mid -1), x, st, en)
        else:
            return last_element(arr, (mid + 1), high, x, st, en)    
    return -1

def is_sorted(a):
    for i in range(len(a)-1):
        if (a[i] > a[i+1]):
            return False
    return True