# Проект 3. Предсказание оценки пользователей Booking.com.

## Оглавление  
[1. Описание проекта](#Описание-проекта)  
[2. Краткая информация о данных](#Краткая-информация-о-данных)  
[3. Этапы работы над проектом](#Этапы-работы-над-проектом)  
[4. Результат](#Результаты)  
[5. Ссылка на работу в кагле](#kaggle-link)

### Описание проекта    
Классный проект, в котором необходимо предсказать оценку пользователей на основании таблицы в 500к строк.

### Краткая информация о данных
Я не буду расписывать столбцы, их там 17. В самом проекте у меня есть их описание.

### Этапы работы над проектом  
Я долго думал над структурой, т.к. если разбирать последовательно, получается каша. И на самом деле я просто увидел, что критическая ошибка лишь увеличивается с каждым моим действием. Потому решил разделить исследование на 2 блока:
1. В первом находится разбор не текстовых признаков. Числовых и там был 1 признак типа список. Я попробовал выделить из этих признаков как можно большее количество информации, пытался масштабировать, группировать, следить за корреляцией, но всё равно результат был печальный. А потом я приступил к разбору текстовых столбцов.

2. Именно они оказались наиболее влияющими на точность обучения модели. Применив 2 алгоритма к каждому из столбцов и выделив ключевые слова в бинарные столбцы, я остался довольным результатом. Считаю, что классно у меня получилось всё это расписать.

Кайфанул короче.)

### Результаты 
В результате у меня получилось не плохое такое обучение модели.<br>
Что позволило мне залететь в <a href='https://www.kaggle.com/competitions/sf-booking/leaderboard'>топ</a>.) 

:arrow_up:[к оглавлению](#Оглавление)

### Kaggle link

[Kaggle link](https://www.kaggle.com/code/akialema/akialema-reviewer-score-prediction/notebook)

---

⬅️ [BACK TO REPO](https://github.com/Akialema/PROJECTS.EDU/tree/main) ⬅️