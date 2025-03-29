# Рассчитаем межквартильный диапазон
Q1 = df['cond'].quantile(0.25)
Q3 = df['cond'].quantile(0.75)
IQR = Q3 - Q1

# Определение границ для выбросов
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Удаление строк с выбросами
data = df[(df['cond'] > lower_bound) & (df['cond'] < upper_bound)]
