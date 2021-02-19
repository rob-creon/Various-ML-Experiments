def do_stock_sim(result_test, params, stock_data, silent=False):
    if not silent:
        print('calculating effectiveness of model...')
    starting_offset = int(0.9 * len(stock_data.adjusted_closes))

    sign_acc = 0
    money = 10000
    shares = 0
    for i in range(len(result_test)):
        index = starting_offset + params.history_points + i
        expected_change = result_test[i]
        today_price = stock_data.adjusted_closes[index-1]
        tomorrow_predicted = today_price + expected_change

        if index == len(stock_data.adjusted_closes):
            if not silent:
                print(f'tomorrows predicted change is {expected_change}, for an expected price of ${tomorrow_predicted}.')
        else:
            predicted = tomorrow_predicted
            actual = stock_data.adjusted_closes[index]
            if i > 0:
                real_change = stock_data.adjusted_closes[index] - stock_data.adjusted_closes[index - 1]
                if (real_change > 0 and expected_change) > 0 or (real_change < 0 and expected_change < 0):
                    sign_correct = True
                    sign_acc += 1
                else:
                    sign_correct = False
                if not silent:
                    print(f'{stock_data.dates_str[index]} predicted {expected_change}, ${predicted}, actual ${actual}', f'sign correct?={sign_correct}, inaccuracy = {expected_change - real_change}')
            else:
                if not silent:
                    print(f'{stock_data.dates_str[index]} predicted {expected_change}, ${predicted}, actual ${actual}')
        if tomorrow_predicted > today_price and money > 0:
            s = money / today_price
            if not silent:
                print(f'buying {s} shares for ${money}')
            shares += s
            money = 0
        if tomorrow_predicted < today_price and shares > 0:
            if not silent:
                print(f'selling {shares} for {shares * today_price}')
            money += shares * today_price
            shares = 0
        if not silent:
            print(f'current value of account is',
              (money + shares * today_price))
            print(f'cash=${money}, shares={shares} (valued at {shares * today_price})')
            print()

    sign_acc /= len(result_test)
    if not silent:
        print(f'sign acc = {sign_acc*100}%')
        print(f'value of account (started with $10,000) is', (money + shares * stock_data.adjusted_closes[-1]))
    return money + shares * stock_data.adjusted_closes[-1]
