
//
// This is example code from Chapter 12.3 "A first example" of
// "Programming -- Principles and Practice Using C++" by Bjarne Stroustrup
//

#include "Simple_window.h"    // get access to our window library
#include "Graph.h"            // get access to our graphics library facilities

//------------------------------------------------------------------------------

int main()
{
    using namespace Graph_lib;   // our graphics facilities are in Graph_lib

    Point tl(100,100);           // to become top left  corner of window

    Simple_window win(tl,600,400,"Luohaonan");    // make a simple window

    Graph_lib::Polygon poly;                // make a shape (a polygon)

    poly.add(Point(300,200));    // add a point
    poly.add(Point(350,100));    // add another point
    poly.add(Point(400,200));    // add a third point 

    poly.set_color(Color::red);  // adjust properties of poly

    win.attach (poly);           // connect poly to the window

    Function sine(sin, 0, 100, Point(20, 150), 1000, 50, 50);
    win.attach(sine);

    Graph_lib::Rectangle r(Point(200, 200), 100, 50);
    win.attach(r);

    Closed_polyline poly_rect;
    poly_rect.add(Point(100, 50));
    poly_rect.add(Point(200, 50));
    poly_rect.add(Point(200, 100));
    poly_rect.add(Point(100, 100));
    poly_rect.add(Point(50, 75));

    r.set_fill_color(Color::yellow);	// color the inside of the rectangle

    poly.set_style(Line_style(Line_style::dash, 4));	  // make the triangle fat

    poly_rect.set_fill_color(Color::green);
    poly_rect.set_style(Line_style(Line_style::dash, 2));

    Text t(Point(100, 100), "Hello, graphical world!");  // add text

    win.wait_for_button();       // give control to the display engine
}

//------------------------------------------------------------------------------
