#%%
class TenisBall:
    def __init__(self):
        self.ball_stack=[]
        self.top = 0
        return self._get_value()
    
    def _get_value(self):
        print('----------------')
        print('1. 테니스 공 넣기')
        print('2. 테니스 공 꺼내기')
        print('3. 테니스 공 개수 출력')
        print('4. 종료')
        
        input_value = input('메뉴를 선택하시오:')
        
        if input_value =='1':
            return self._push()
        elif input_value == '2':
            return self._pop()
        elif input_value == '3':
            return self._show()
        elif input_value =='4':
            return
        else:
            print('잘못된 입력입니다.')
            return self._get_value()

    def _push(self):
        if len(self.ball_stack) ==5:
            print('케이스가 꽉 찼습니다')
            return self._get_value()
        
        self.top+=1
        self.ball_stack.append(self.top)
        
        return self._show()

    def _pop(self):
        if len(self.ball_stack)==0:
            print('케이스가 비어 있습니다')
            return self._get_value()

        print(f'Pop({self.top})')
        del self.ball_stack[-1]
        self.top-=1
        
        return self._show()


    def _show(self):
        if len(self.ball_stack)==0:
            print('케이스가 비어있습니다')
            return self._get_value()            
        print('[공의 개수]:',len(self.ball_stack))
        for ball in self.ball_stack[::-1]:
            print(ball,end=' ')
        print('')
        
        return self._get_value()


test = TenisBall()
