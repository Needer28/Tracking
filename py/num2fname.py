def num2fname(num):
    """ Turns an int num into MOT image file name.
        Example: input 32, output '000032.jpg'
    """

    
    if num < 10 and num > 0:
        return '00000' + str(num) + '.jpg'
    elif num < 100:
        return '0000' + str(num) + '.jpg'
    elif num < 1000:
        return '000' + str(num) + '.jpg'
    else:
        return '00' + str(num) + '.jpg'
