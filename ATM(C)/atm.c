#include <stdio.h>
#include <ncurses.h>
#include <unistd.h>
void drawBackground(int colorPair) {
    start_color();
    init_pair(1, COLOR_RED, COLOR_BLACK);
    init_pair(2, COLOR_GREEN, COLOR_BLACK);
    attron(COLOR_PAIR(colorPair));
    mvprintw(0, 0, "ATM Interface");
    attroff(COLOR_PAIR(colorPair));
    refresh();}
// Define ANSI color escape sequences
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_RESET   "\x1b[0m"
typedef struct {
    long int account_number;
    int password;
} acc_no;
int bal;
int p;
int account(long int x,int cardpin) {
    acc_no acc[10];
    acc[0].account_number = 123456789;
    acc[1].account_number = 123456788;
    acc[2].account_number = 123456787;
    acc[3].account_number = 123456786;
    acc[4].account_number = 123456785;
    acc[5].account_number = 123456784;
    acc[6].account_number = 123456783;
    acc[7].account_number = 123456782;
    acc[8].account_number = 123456781;
    acc[9].account_number = 123456780;
    acc[0].password = 1234;
    acc[1].password = 1235;
    acc[2].password = 1236;
    acc[3].password = 1237;
    acc[4].password = 1238;
    acc[5].password = 1239;
    acc[6].password = 1241;
    acc[7].password = 1242;
    acc[8].password = 1243;
    acc[9].password = 1244; 
    if (x == acc[0].account_number&& cardpin==acc[0].password){
        p=0;
        bal=10000;
        return 1;}
           else if (x == acc[2].account_number&& cardpin==acc[2].password){
        p=2;
        bal=120000;
        return 1;}   
       else if (x == acc[1].account_number&& cardpin==acc[1].password){
        p=1;
        bal=130000;
        return 1;}    
       else if (x == acc[3].account_number&& cardpin==acc[3].password){
        p=3;
        bal=14000;
        return 1;}   
       else if (x == acc[4].account_number&& cardpin==acc[4].password){
        p=4;
        bal=12500;
        return 1;}    
       else if (x == acc[5].account_number&& cardpin==acc[5].password){
        p=5;
        bal=13550;
        return 1;}    
       else if (x == acc[6].account_number&& cardpin==acc[6].password){
        p=6;
        bal=11100;
        return 1;}   
       else if (x == acc[7].account_number&& cardpin==acc[7].password){
        p=7;
        bal=12345;
        return 1;}    
        else if (x == acc[8].account_number&& cardpin==acc[8].password){
        p=8;
        bal=13245;
        return 1;}    
       else if (x == acc[9].account_number&& cardpin==acc[9].password){
        p=9;
        bal=12346;
        return 1;}    
    else {
       return 0;} }
       void withdrawal(int enteramount, int amount) {
    // Input validation
    if (enteramount < 0 || amount < 0) {
        printf("Invalid input. Amounts should be non-negative.\n");
        return;
    }
    int note500 = 0, note100 = 0, note50 = 0, note20 = 0, note10 = 0, note5 = 0, note2 = 0, note1 = 0;
    if (amount >= 500) {
        note500 = amount / 500;
        amount -= note500 * 500;
    }
    if (amount >= 100) {
        note100 = amount / 100;
        amount -= note100 * 100;
    }
    if (amount >= 50) {
        note50 = amount / 50;
        amount -= note50 * 50;
    }
    if (amount >= 20) {
        note20 = amount / 20;
        amount -= note20 * 20;
    }
    if (amount >= 10) {
        note10 = amount / 10;
        amount -= note10 * 10;
    }
    if (amount >= 5) {
        note5 = amount / 5;
        amount -= note5 * 5;
    }
    if (amount >= 2) {
        note2 = amount / 2;
        amount -= note2 * 2;
    }
    if (amount >= 1) {
        note1 = amount;
    }

    // Print required notes
    printf("Total number of notes:\n");
    printf("500 = %d\n", note500);
    printf("100 = %d\n", note100);
    printf("50 = %d\n", note50);
    printf("20 = %d\n", note20);
    printf("10 = %d\n", note10);
    printf("5 = %d\n", note5);
    printf("2 = %d\n", note2);
    printf("1 = %d\n", note1);
        for (int i = 0; i < note500; i++) {
        printf("*****   *****    ***** \n");
        printf("*      *     *  *     *\n");
        printf("*      *     *  *     *\n");
        printf("*****  *     *  *     *\n");
        printf("    *  *     *  *     *\n");
        printf("    *  *     *  *     *\n");
        printf("*****   *****    *****\n\n");
    }
    for (int i = 0; i < note100; i++) {
        printf("  ***    *****    ***** \n");
        printf(" *  *   *     *  *     *\n");
        printf(" *  *   *     *  *     *\n");
        printf("    *   *     *  *     *\n");
        printf("    *   *     *  *     *\n");
        printf("    *   *     *  *     *\n");
        printf(" ******  *****    *****\n\n"); }
    for (int i = 0; i < note50; i++) {
        printf("*****   *****\n");
        printf("*      *     *\n");
        printf("*      *     *\n");
        printf("*****  *     *\n");
        printf("    *  *     *\n");
        printf("    *  *     *\n");
        printf("*****   *****\n"); }
                    for (int i = 0; i < note20; i++) {
        printf("*****    *****\n");
        printf("    *   *     *\n");
        printf("    *   *     *\n");
        printf("*****   *     *\n");
        printf("*       *     *\n");
        printf("*       *     *\n");
        printf("******  *****\n\n"); }
            for (int i = 0; i < note10; i++) {
        printf("  ***    *****\n");
        printf(" *  *   *     *\n");
        printf(" *  *   *     *\n");
        printf("    *   *     *\n");
        printf("    *   *     *\n");
        printf("    *   *     *\n");
        printf(" ******  *****\n"); }
            for (int i = 0; i < note5; i++) {
        printf("*****\n");
        printf("*\n");
        printf("*\n");
        printf("*****\n");
        printf("    *\n");
        printf("    *\n");
        printf("*****\n"); }
     for (int i = 0; i < note2; i++) {
        printf("*****\n");
        printf("    *\n");
        printf("    *\n");
        printf("*****\n");
        printf("*\n");
        printf("*\n");
        printf("******\n\n"); }
            for (int i = 0; i < note1; i++) {
        printf("  ***\n");
        printf(" *  *\n");
        printf(" *  *\n");
        printf("    *\n");
        printf("    *\n");
        printf("    *\n");
        printf(" ******\n"); }
                         printf("Collect your cash ...\n");
                    }
  void person1(int x,int cardpin,int y,int z,int enteramount,int totalamount,int balance,int b,int a,int accountno,int amount) {
    printf("chose your language \n");
    printf("\n");
    sleep(2);
    printf("enter 1 for ENGLISH \nenter 2 for HINDI");
    printf("\n");
    scanf("%d",&x);
    if (x == 1) {
            printf("choose your banking option \n");
     printf("\n");
     printf("enter 1 for WITHDRAWL,\nenter 2 for BANK STATEMENT,\nenter 3 for BALANCE,\nenter 4 for QUICK CASH,\nenter 5 for FAST CASH,\nenter 6 for SERVICE,\nenter 7 for ACCOUNT TRANSFER \n");
       printf("\n");
       scanf("%d",&y);
            if (y == 1) {
                printf("choose you withdrawl account \n");
                printf("\n");
                sleep(1);
     printf("enter 1 for savings account,enter 2 for current account \n");
     printf("\n");
               scanf("%d",&z);
                if (z == 1) {
                    totalamount = bal;
                    printf("totalamount=%d \n",totalamount);
                    printf("\n");
                    printf("enter amount \n");
                     scanf("%d",& enteramount);
                    if (totalamount > enteramount) {
                         balance = totalamount - enteramount;
                        printf("balance amount is %d",balance);
                        printf("\n");
                        amount=enteramount;
                        withdrawal(enteramount,amount); 
                        printf("Collect your cash ...\n");}
                     else {
                         printf("amount insufficient \n");
                         printf("\n");}}
                } if (z==2){
                    totalamount = bal;
                 printf("totalamount in your account is %d \n",totalamount);
                     printf("\n");
                     printf("enter amount \n");
                     scanf("%d",&enteramount);
                    if (totalamount > enteramount) {
                         balance = totalamount - enteramount;
                         printf("balance amount is %d \n",balance);
                         printf("\n");
                                                 amount=enteramount;
                       withdrawal(enteramount,amount);
                         printf("Collect your cash ...\n");
                    } else {
                         printf("amount insufficient \n");
                         printf("\n");
                         printf("Remove your card...");
                    } }
                if (y == 2) {
                    switch(p){
                        case 0: 
                         printf("Harsha Vardhan\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
                    case 1:
                     printf("Leekith\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
                    case 2:
                     printf("Mani vignesh\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
                    case 3:
                     printf("Rajesh\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
                    case 4:
                     printf("Shriniketan\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
                    case 5:
                     printf("Nalini mam\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
                    case 6:
                     printf("Kritish\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
                    case 7:
                     printf("Snehin\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
                    case 8:
                     printf("Akhil\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
                    case 9:
                     printf("Rakesh\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
                    
                } }
                    if (y == 3) {
                         printf("balance in your account is %d \n",bal);
                    } 
                        if (y == 4) {
                             printf("amount to take in Quick Cash is 15000");
                             printf("\n");
                             printf("enter 1 if yes\n enter 2 if no ");
                             scanf("%d",&b);
                             b =x;
                            if (b == 1) {
                                for (int i=1;i<8;i++) {
      printf("*****    *****   *****   *****\n");
        printf("    *   *     * *     * *     *\n");
        printf("    *   *     * *     * *     *\n");
        printf("*****   *     * *     * *     *\n");
        printf("*       *     * *     * *     *\n");
        printf("*       *     * *     * *     *\n");
        printf("******   *****   *****   *****\n\n");
}
for (int i=1;i<3;i++) {
                printf("*****   *****    ***** \n");
        printf("*      *     *  *     *\n");
        printf("*      *     *  *     *\n");
        printf("*****  *     *  *     *\n");
        printf("    *  *     *  *     *\n");
        printf("    *  *     *  *     *\n");
        printf("*****   *****    *****\n\n"); }
        printf("collect 15000rs...\n");
                                 
                            } else {
                                 printf("Remove your card...\n");
                                
                            }} 
                            if (y == 5) {
                                 printf("Fast Cash\n");
                                 printf("enter 1for 100rs,\nenter 2 for 200rs,\nenter 3 for 500rs,\nenter 4 for 2000rs,\n enter 5 for 5000rs,\n entre 6 for 10000rs\n");
                               scanf("%d",&a);
                                if (a == 1) {
                                             printf("  ***    *****    ***** \n");
                                             printf(" *  *   *     *  *     *\n");
                                             printf(" *  *   *     *  *     *\n");
                                             printf("    *   *     *  *     *\n");
                                             printf("    *   *     *  *     *\n");
                                             printf("    *   *     *  *     *\n");
                                             printf(" ******  *****    *****\n\n");
                                             printf("collect your 100rs..\n");
                                     
                                } 
                                    if (a == 2) {
        printf("*****    *****   *****\n");
        printf("    *   *     * *     *\n");
        printf("    *   *     * *     *\n");
        printf("*****   *     * *     *\n");
        printf("*       *     * *     *\n");
        printf("*       *     * *     *\n");
        printf("******   *****   *****\n\n");
                                         printf("collect your 200rs..\n");
                                    }
                                        if (a == 3) {
                                                    printf("*****   *****    ***** \n");
                                                    printf("*      *     *  *     *\n");
                                                    printf("*      *     *  *     *\n");
                                                    printf("*****  *     *  *     *\n");
                                                    printf("    *  *     *  *     *\n");
                                                    printf("    *  *     *  *     *\n");
                                                    printf("*****   *****    *****\n\n");
                                                    printf("collect your 500rs..\n");
                                        } 
                                            if (a == 4) {
        printf("*****    *****   *****   *****\n");
        printf("    *   *     * *     * *     *\n");
        printf("    *   *     * *     * *     *\n");
        printf("*****   *     * *     * *     *\n");
        printf("*       *     * *     * *     *\n");
        printf("*       *     * *     * *     *\n");
        printf("******   *****   *****   *****\n\n");
                                                 printf("collect your 2000r..s\n");
                                            } 
                                                if (a == 5) {
                                                    for (int i=1;i<3;i++){
                                                    
        printf("*****    *****   *****   *****\n");
        printf("    *   *     * *     * *     *\n");
        printf("    *   *     * *     * *     *\n");
        printf("*****   *     * *     * *     *\n");
        printf("*       *     * *     * *     *\n");
        printf("*       *     * *     * *     *\n");
        printf("******   *****   *****   *****\n\n"); }
        for (int i=1;i<3;i++){
         printf("*****   *****    ***** \n");
          printf("*      *     *  *     *\n");
           printf("*      *     *  *     *\n");
                                                    printf("*****  *     *  *     *\n");
                                                    printf("    *  *     *  *     *\n");
                                                    printf("    *  *     *  *     *\n");
                                                    printf("*****   *****    *****\n\n");}
                                                printf("collect your 5000rs..\n");
                                                } if (a==6){
                                                    for (int i=1;i<6;i++){
        printf("*****    *****   *****   *****\n");
        printf("    *   *     * *     * *     *\n");
        printf("    *   *     * *     * *     *\n");
        printf("*****   *     * *     * *     *\n");
        printf("*       *     * *     * *     *\n");
        printf("*       *     * *     * *     *\n");
        printf("******   *****   *****   *****\n\n");}
                                                     printf("collect your 10000rs..\n");
                                }else{
                                 printf("remove your card..\n");}}
              
                 if (y == 6) {
                     printf("press \n1-Help line\n2-cheque book request \n3-credit card application\n");
                    scanf("%d",&z);
                    switch(z) {
                        case 1:
                        printf("please contact your bank branch\n");
                        break;
                        case 2:
                        printf("Do you want to request cheque book on the same Account number\npress 1-yes \t 2-no\n");
                        scanf("%d",&a);
                        if (a==1){
                            printf("Your cheque book request is approved,You may get further information within 2 days\n");
                        }else{
                            printf("Please consult your respective bank for request\n");
                        }
                        break;
                        case 3:
                                                printf("Do you want to request credit card  on the same Account number\npress 1-yes \t 2-no\n");
                        scanf("%d",&a);
                        if (a==1){
                            printf("Your cheque book request is approved,You may get further information within 2 days\n");
                        }else{
                            printf("Please consult your respective bank for request\n");
                        }
                    }
                    
                    } if(y==7) {
            totalamount = 12000;
                printf("enter bank account holder account no\n");
                 scanf("%d",&accountno);
                     if (accountno==123456789) {
                    printf ("Account holder name is Harsha Vardhan ");}
                                          else if (accountno==123456789) {
                     printf("Account holder name is leekith ");}
                                          else if (accountno==123456789) {
                     printf("Account holder name is Mani vignesh"); }
                                          else if (accountno==123456789) {
                     printf("Account holder name is Rajesh");}
                                          else if (accountno==123456789) {
                     printf("Account holder name is Shriniketan ");}
                                          else if (accountno== 123456789) {
                     printf("Account holder name is Nalini mam");}
                                          else if (accountno== 123456789) {
                     printf("Account holder name is Snehin ");}
                                          else if (accountno==123456789) {
                     printf("Account holder name is Akhil"); }
                                          else if (accountno== 123456789) {
                     printf("Account holder name is Rakesh ");}
                 printf("enter amount in rs\n");
                scanf("%d",&enteramount);
                 printf("balance amount is %d\n",totalamount-enteramount);
                printf("bank require of minimum 3hrs of time to transfer\n"); }  }
 else {
             printf("अपना बैंकिंग विकल्प चुनें");
             printf("\n");
             printf("निकासी के लिए 1 दर्ज करें, \n बैंक स्टेटमेंट के लिए 2 दर्ज करें, \nशेष राशि के लिए 3 दर्ज करें, \nत्वरित नकदी के लिए 4 दर्ज करें,\n त्वरित नकदी के लिए 5 दर्ज करें\n, सेवा के लिए 6 दर्ज करें");
            scanf("%d",&y);
            if (y == 1) {
                 printf("अपना निकासी खाता चुनें");
                 printf("\n");
                 sleep(1);
                 printf("बचत खाते के लिए 1 दर्ज करें, चालू खाते के लिए 2 दर्ज करें");
            printf("\n");
            scanf("%d",&z);
                if (z == 1) {
                    totalamount = bal;
                     printf("कुलराशि=%d\n ",totalamount);
                     printf("\n");
                     printf("enter amount");
                     scanf("%d",&enteramount);
                    if (totalamount > enteramount) {
                         balance = totalamount - enteramount;
                         printf("balance amount is %d", balance);
                         withdrawal(enteramount,amount);
                         printf("अपना नकद इकट्ठा करें");
                         }
                     else {
                         printf("राशि अपर्याप्त");
                    } }
                 if (z==2) {
                     totalamount=bal;
                    printf("कुलराशि=%d\n ",totalamount);
                    printf("\n");
                     printf("enter amount \n");
                     scanf("%d",&enteramount);
                    if (enteramount < totalamount) {
                         balance = totalamount - enteramount;
                         printf("balance amount is%d \n",balance);
                         printf("\n");
                          amount=enteramount;
withdrawal(enteramount,amount);
                         printf("अपना नकद इकट्ठा करें");
                    } else {
                         printf("राशि अपर्याप्त");
                    }
                }}
             
                if (y == 2) {
          switch(p){
                        case 0: 
                         printf("Harsha Vardhan\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
                    case 1:
                     printf("Leekith\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
                    case 2:
                     printf("Mani vignesh\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
                    case 3:
                     printf("Rajesh\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
                    case 4:
                     printf("Shriniketan\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
                    case 5:
                     printf("Nalini mam\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
                    case 6:
                     printf("Kritish\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
                    case 7:
                     printf("Snehin\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
                    case 8:
                     printf("Akhil\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
                    case 9:
                     printf("Rakesh\taccont summary\nopeningbalance=51,432\nwithdrawls=12,876\ndeposit=43,876\n");
                    printf("Here is your recipt... \n");
                    break;
          }
                } 
                    if (y == 3) {
                         printf("आपके खाते में शेष राशि %d है\n",bal);
                    } 
                        if (y == 4) {
                             printf("amount to take in Quick Cash is 15000 \n");
                             printf("enter 1 if yes \n,enter 2 if no \n");
                             b = x;
                            if (b == 1) {
                                for(int i=1;i<8;i++){
      printf("*****    *****   *****   *****\n");
        printf("    *   *     * *     * *     *\n");
        printf("    *   *     * *     * *     *\n");
        printf("*****   *     * *     * *     *\n");
        printf("*       *     * *     * *     *\n");
        printf("*       *     * *     * *     *\n");
        printf("******   *****   *****   *****\n\n");}
        for (int i=1;i<3;i++) {
                printf("*****   *****    ***** \n");
        printf("*      *     *  *     *\n");
        printf("*      *     *  *     *\n");
        printf("*****  *     *  *     *\n");
        printf("    *  *     *  *     *\n");
        printf("    *  *     *  *     *\n");
        printf("*****   *****    *****\n\n"); }
                                 printf("collect 15000rs");
                            } else {
                                 printf("अपना कार्ड निकालें...");
                            }
                        } else {
                            if (y == 5) {
                                printf("तेजी से नकदी\n");
                                 printf("100 रुपये के लिए 1 दर्ज करें\n, 200 रुपये के लिए 2 दर्ज करें\n, 500 रुपये के लिए 3 दर्ज करें\n, 2000 रुपये के लिए 4 दर्ज करें\n, 5000 रुपये के लिए 5 दर्ज करें\n, 10000 रुपये के लिए 6 दर्ज करें\n");
                                 scanf("%d",&a);
                                if (a == 1) {
                                                                                 printf("  ***    *****    ***** \n");
                                             printf(" *  *   *     *  *     *\n");
                                             printf(" *  *   *     *  *     *\n");
                                             printf("    *   *     *  *     *\n");
                                             printf("    *   *     *  *     *\n");
                                             printf("    *   *     *  *     *\n");
                                             printf(" ******  *****    *****\n\n");
                                     printf("अपने 100 रुपये इकट्ठा करें");
                                } 
                                    if (a == 2) {
                                                printf("*****    *****   *****\n");
        printf("    *   *     * *     *\n");
        printf("    *   *     * *     *\n");
        printf("*****   *     * *     *\n");
        printf("*       *     * *     *\n");
        printf("*       *     * *     *\n");
        printf("******   *****   *****\n\n");
                                         printf("अपने 200 रुपये इकट्ठा करें");
                                    }
                                        if (a == 3) {
                                                                                                printf("*****   *****    ***** \n");
                                                    printf("*      *     *  *     *\n");
                                                    printf("*      *     *  *     *\n");
                                                    printf("*****  *     *  *     *\n");
                                                    printf("    *  *     *  *     *\n");
                                                    printf("    *  *     *  *     *\n");
                                                    printf("*****   *****    *****\n\n");
                                             printf("अपने 500 रुपये इकट्ठा करें");
                                        } 
                                            if (a == 4) {
                                                        printf("*****    *****   *****   *****\n");
        printf("    *   *     * *     * *     *\n");
        printf("    *   *     * *     * *     *\n");
        printf("*****   *     * *     * *     *\n");
        printf("*       *     * *     * *     *\n");
        printf("*       *     * *     * *     *\n");
        printf("******   *****   *****   *****\n\n");
                                                 printf("अपने 2000 रुपये इकट्ठा करें");
                                            } else {
                                                if (a == 5) {
                                                    for (int i=1;i=2;i++){
                                                            printf("*****    *****   *****   *****\n");
        printf("    *   *     * *     * *     *\n");
        printf("    *   *     * *     * *     *\n");
        printf("*****   *     * *     * *     *\n");
        printf("*       *     * *     * *     *\n");
        printf("*       *     * *     * *     *\n");
        printf("******   *****   *****   *****\n\n"); }  
        for (int i=1;i=2;i++){
         printf("*****   *****    ***** \n");
          printf("*      *     *  *     *\n");
           printf("*      *     *  *     *\n");
                                                    printf("*****  *     *  *     *\n");
                                                    printf("    *  *     *  *     *\n");
                                                    printf("    *  *     *  *     *\n");
                                                    printf("*****   *****    *****\n\n"); }
                                                     printf("अपने 5000 रुपये इकट्ठा करें");
                                                } if (a==6) {
                                                    for (int i=1;i<6;i++){
                                                            printf("*****    *****   *****   *****\n");
        printf("    *   *     * *     * *     *\n");
        printf("    *   *     * *     * *     *\n");
        printf("*****   *     * *     * *     *\n");
        printf("*       *     * *     * *     *\n");
        printf("*       *     * *     * *     *\n");
        printf("******   *****   *****   *****\n\n");}
                                                     printf("अपने 10000 रुपये इकट्ठा करें");
                                 }} }else{
                                     printf("अपना कार्ड हटाओ");
}                            if (y==6){
                                 printf("सेवा");
                                 printf("\n");
 printf("press \n1-Help line\n2-cheque book request \n3-credit card application\n");
                    scanf("%d",&z);
                    switch(z) {
                        case 1:
                        printf("कृपया अपनी बैंक शाखा से संपर्क करें\n");
                        break;
                        case 2:
                        printf("क्या आप उसी खाता संख्या पर चेक बुक का अनुरोध करना चाहते हैं?\npress 1-yes \t 2-no\n");
                        scanf("%d",&a);
                        if (a==1){
                            printf("आपका चेक बुक अनुरोध स्वीकृत है, आपको 2 दिनों के भीतर अधिक जानकारी मिल सकती है\n");
                        }else{
                            printf("कृपया अनुरोध के लिए अपने संबंधित बैंक से परामर्श लें\n");
                        }
                        break;
                        case 3:
                                                printf("क्या आप उसी खाता संख्या पर क्रेडिट कार्ड का अनुरोध करना चाहते हैं?\npress 1-yes \t 2-no\n");
                        scanf("%d",&a);
                        if (a==1){
                            printf("आपका चेक बुक अनुरोध स्वीकृत है, आपको 2 दिनों के भीतर अधिक जानकारी मिल सकती है\n");
                        }else{
                            printf("कृपया अनुरोध के लिए अपने संबंधित बैंक से परामर्श लें\n");
                        }
                    }
                        }if(y==7) {
            totalamount = 120000;
                printf("बैंक खाताधारक खाता संख्या दर्ज करें\n");
                 scanf("%d",&accountno);
                 printf("राशि रुपये में दर्ज करें\n");
                scanf("%d",&enteramount);
                 printf("शेष राशि है %d\n",totalamount-enteramount);
                printf("बैंक को स्थानांतरण के लिए न्यूनतम 3 घंटे का समय चाहिए\n");
                                }
                        }
        } 
    }
int main () {
     int x=0;
     int cardpin=0;
     int y=0;
     int z=0;
     int enteramount=0;
     int totalamount=0;
     int balance=0;
     int b=0;
     int a=0;
     int accountno=0;
     int amount=0;
     long int ex;
     long int xe;
     int aa;
         initscr(); // Initialize the library
    curs_set(0); // Hide the cursor
    for (int i = 0; i < 10; ++i) {
        clear(); // Clear the screen
        // Draw the background with alternating colors
        if (i % 2 == 0) {
            drawBackground(1); // Red background
        } else {
            drawBackground(2); // Green background
        }
        usleep(500000); // Sleep for a short duration to create the effect 
        }
    endwin(); // End the library
printf(".\n");
printf(ANSI_COLOR_RED   "    *    ********  **      **\n"ANSI_COLOR_RESET);
printf(ANSI_COLOR_RED   "   * *      *      *  *   * *\n"ANSI_COLOR_RESET);
printf(ANSI_COLOR_GREEN "  *****     *      *    *   *\n"ANSI_COLOR_RESET);
printf(ANSI_COLOR_YELLOW " *     *    *      *        *\n"ANSI_COLOR_RESET);
printf(ANSI_COLOR_YELLOW "*       *   *      *        *\n\n"ANSI_COLOR_RESET);
sleep(2);
printf(ANSI_COLOR_BLUE"****WELCOME**** \n"ANSI_COLOR_RESET);
printf("\n");
sleep(2);
printf("Insert your card.. \n");
printf("\n");
sleep(2);
printf("Please do not remove your chipcard leave chipcard inserted during the entire transcation \n");
printf("\n");
sleep(4);
printf("Enter your 10 digit Account number \n");
printf("\n");
  scanf("%ld",&ex);
printf("please enter your card pin \n");
printf("\n");
scanf("%d",&cardpin);
 aa= account(ex,cardpin);
if (aa==1) {
person1(x,cardpin,y,z,enteramount,totalamount,balance,b,a,accountno,amount);} 
else{
printf("no card holder with given account number and card pin...\n");} 
return 0; }


