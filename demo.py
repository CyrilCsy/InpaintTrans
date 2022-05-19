def fun(x=None):
    x = 1
    y = 2
    z = 3
    return {'x':x, 'y':y, 'z':z}

if __name__ == "__main__":
    f = fun()
    print(f.get('x'))
    x = 111
    print('qwe'+str(x)+'qwe')